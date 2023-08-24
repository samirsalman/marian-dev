#pragma once

#include "common/options.h"
#include "common/signal_handling.h"
#include "training/training_state.h"
#include "training/validator.h"
#include "training/communicator.h"
#include "layers/loss.h"

namespace marian {

/**
 * This exception gets thrown when a training run divergence was detected. See below in main update function.
*/
class DivergenceException : public std::runtime_error {
public:
  DivergenceException(float averageSlow, float averageFast, float sigmas)
  : std::runtime_error(fmt::format(
      "Detected training divergence: slow-moving average loss {:.4f} exceeded by fast-moving average loss {:.4f} by {:.4f} = {:.4f} * sigmas", 
      averageSlow, averageFast, averageFast - averageSlow, sigmas)) 
    {}
};

class Scheduler : public TrainingObserver {
private:
  Ptr<Options> options_;
  Ptr<TrainingState> state_;
  std::vector<Ptr<ValidatorBase>> validators_;
  Ptr<IMPIWrapper> mpi_;

  bool first_{true};                  // true if this is the first update after renewing the training

  bool throwOnDivergence_{false};   // throw an exception if training divergence is detected
  size_t lossAvgWindowSlow_{1000};  // window size for slow-moving average loss for divergence detection
  size_t lossAvgWindowFast_{10};    // window size for fast-moving average loss for divergence detection
  float divergenceTolerance_{5.f};  // tolerance for divergence detection as multiples of standard deviation
  SchedulingParameter throwAfter_;  // for diagnostics only; training will throw if non-zero and training has progressed this far
  
  size_t gradientNormAvgWindow_{100}; // window size for recording the exponential average of gradient norms, after this many updates about 90% of the mass comes from this many last updates
  SchedulingParameter logicalEpoch_;
  size_t logicalEpochWidth_{0};

  timer::Timer timer_;
  timer::Timer heartBeatTimer_;

  // The variable helps to keep track of the end of the current epoch
  // (regardless if it's the 1st or nth epoch and if it's a new or continued training),
  // which indicates the end of the training data stream from STDIN
  bool endOfStdin_{false};  // true at the end of the epoch if training from STDIN;

  // @TODO: figure out how to compute this with regard to updates as well, although maybe harder since no final value
  // determine scheduled LR decay factor (--lr-decay-inv-sqrt option)
  float getScheduledLRDecayFactor(const TrainingState& state) const {
    auto args = options_->get<std::vector<std::string>>("lr-decay-inv-sqrt");
    ABORT_IF(args.empty() || args.size() > 2, "--lr-decay-inv-sqrt argument must be one or two numbers with units");
    auto decayGoogle = SchedulingParameter::parse(args[0]);
    size_t progress = state.getProgressIn(decayGoogle.unit);
    size_t start = decayGoogle.n;
    if (args.size() > 1) {
      auto decayStart = SchedulingParameter::parse(args[1]);
      ABORT_IF(decayStart && decayStart.unit != decayGoogle.unit,
               "both --lr-decay-inv-sqrt arguments must have the same unit");
      start = decayStart.n;
    }
    if (decayGoogle && progress > start) {
      progress = progress - start + decayGoogle.n; // shift so that we get 1 at progress==start
      return (float)(std::sqrt((double)decayGoogle.n / (double)progress));
    }
    else
      return 1.f;
  }

  void updateLearningRate(TrainingState& state) const {
    float baselr = options_->get<float>("learn-rate");

    // warm-up factor
    float warmupFactor = 1.f;
    auto warmupParam = SchedulingParameter::parse(options_->get<std::string>("lr-warmup"));
    if(warmupParam) {
      ABORT_IF(state.warmupStart && state.warmupStart.unit != warmupParam.unit,
               "lr-warmup and warmup-start must have the same unit");
      auto bno = state.getProgressIn(warmupParam.unit) - state.warmupStart.n;
      warmupFactor = std::min(1.f, (float)bno / (float)warmupParam.n);
    }

    // TODO: why lr-warmup-start-rate is extracted from options_ instead of using state.warmupStart?
    float lrStart = options_->get<float>("lr-warmup-start-rate");
    baselr = lrStart + (baselr - lrStart) * warmupFactor; // linear interpolation between
                                                          // lr-warmup-start-rate to learn-rate

    // schedule-based decay factor (--lr-decay-inv-sqrt)
    float scheduledDecayFactor = getScheduledLRDecayFactor(state);
    baselr = baselr * scheduledDecayFactor;

    // factor in state-based decay and set final LR as state.eta
    state.updateEta(baselr);
  }

  std::string formatLoss(std::string lossType,
                         bool dispLabelCounts,
                         size_t batchLabels,
                         Ptr<TrainingState> state) {
    std::stringstream ss;
    ss << "Cost ";
    ss << std::setprecision(8) << std::fixed;

    // @TODO: put a single loss formatting function into loss.h and reuse here to avoid code duplication
    // @TODO: use dispLabelCounts with any display type?
    // @TODO: bugbug cost-type ce-mean-words with multi-loss-type mean divides too much in display
    if(lossType == "ce-mean-words") {
      ss << state->costSum / state->costCount;
    } else if(lossType == "ce-sum" && dispLabelCounts) {
      ss << state->costSum / state->costCount
         << " * " << utils::withCommas((size_t)state->costCount);
      if(batchLabels > 0)
         ss << " @ " << utils::withCommas(batchLabels);
      ss << " after " << utils::withCommas(state->labelsTotal);
    } else if(lossType == "ce-sum" && !dispLabelCounts) {
      ss << state->costSum / state->updatesDisp; // average over batches
    } else if(lossType == "perplexity") {
      ss << std::exp(state->costSum / state->costCount);
    } else if(lossType == "cross-entropy" || lossType == "ce-mean") { // backwards-compat, @TODO: get rid of this?
      ss << state->costSum / state->samplesDisp;
    } else {
      ABORT("Unknown loss type {}", lossType);
    }

    return ss.str();
  }

  // Here we calculate the logical epoch as defined by the user, by default this will be just a traditional data epoch.
  // We understand a data epoch as a complete pass throught the training data as far as that information is available.
  // By contrast, a logical epoch is defined somewhat indepdently of the number of data passes as by the number of seen updates or labels
  // or as a multitude of data epochs.
  float calculateLogicalEpoch() {
    if(logicalEpoch_.unit == SchedulingUnit::epochs)
      return (float)state_->epochs / (float)logicalEpoch_.n;      // logical epoch as multiple of n data epochs
    else if(logicalEpoch_.unit == SchedulingUnit::trgLabels)
      return (float)state_->labelsTotal / (float)logicalEpoch_.n; // logical epoch as multiple of n labels
    else if(logicalEpoch_.unit == SchedulingUnit::updates)
      return (float)state_->batches / (float)logicalEpoch_.n;     // logical epoch as multiple of n gradient updates (not actually batches @TODO: change name)
    else
      ABORT("Unknown scheduling unit occurred in logical epoch"); // shouldn't really happen unless we add a new unit in the corresponding enum
  }

  // Formatting for logical epochs
  std::string formatLogicalEpoch() {
    return fmt::format("{:." + std::to_string(logicalEpochWidth_) + "f}", calculateLogicalEpoch());
  }

public:
  Scheduler(Ptr<Options> options, Ptr<TrainingState> state, Ptr<IMPIWrapper> mpi = nullptr)
      : options_(options), state_(state), mpi_(mpi),
        gradientNormAvgWindow_(options_->get<size_t>("gradient-norm-average-window", 100)) {

    auto throwParameters = options_->get<std::vector<std::string>>("throw-on-divergence");
    if(!throwParameters.empty()) {
      throwOnDivergence_ = true;
      if(throwParameters.size() > 0)
        lossAvgWindowSlow_ = std::stoul(throwParameters[0]);
      if(throwParameters.size() > 1)
        lossAvgWindowFast_ = std::stoul(throwParameters[1]);
      if(throwParameters.size() > 2)
        divergenceTolerance_ = std::stof(throwParameters[2]);
      if(throwParameters.size() > 3)
        throwAfter_ = SchedulingParameter::parse(throwParameters[3]);
        
      LOG(info, 
          "[scheduler] Divergence detection is enabled for slow-moving averaging window over {} steps "
          "vs fast-moving window over {} steps with tolerance of {} sigmas", 
          lossAvgWindowSlow_, lossAvgWindowFast_, divergenceTolerance_);

      if(throwAfter_) {
        LOG(warn, "[scheduler] Diagnostic DivergenceException will be thrown when training reaches {}", (std::string)throwAfter_);
      }
    }

    // parse logical-epoch parameters
    auto logicalEpochStr = options->get<std::vector<std::string>>("logical-epoch", {"1e", "0"});
    ABORT_IF(logicalEpochStr.empty(), "Logical epoch information is missing?");

    logicalEpoch_ = SchedulingParameter::parse(logicalEpochStr[0]);

    // here we deduce the floating point width to be used in formatLogicalEpoch()
    if(logicalEpochStr.size() > 1) { // if the width is given, just use that
      logicalEpochWidth_ = std::stoul(logicalEpochStr[1]);
    } else { // the width is not given so we deduce a suitable display width
      if(logicalEpoch_.unit == SchedulingUnit::epochs && logicalEpoch_.n == 1)
        logicalEpochWidth_ = 0; // for a data epoch, output is an integer and looks like before this feature was introduced
      else
        logicalEpochWidth_ = 3; // all other outputs can be fractional, hence floating point format. We choose
                                // 3 as a default which corresponds to a multiplier of 1000 (3 orders of magnitude).
    }

    ABORT_IF(state_->factor != 1, "state.factor unexpectedly not 1 at this point??");
    updateLearningRate(*state);
  }

  // test if any parameters specify dynamic MB scaling
  bool isDynamicMBSizeScaling() const {
    auto mbWarmup = SchedulingParameter::parse(options_->get<std::string>("mini-batch-warmup"));
    auto mbTracking = options_->get<bool>("mini-batch-track-lr");
    return mbWarmup || mbTracking;
  }

  // determine dynamic MB scaling factor
  double getDynamicMBSizeMultiplier() const {
    double ratio = 1.0;

    auto mbWarmup = SchedulingParameter::parse(options_->get<std::string>("mini-batch-warmup"));
    if (mbWarmup) {
      // mini-batch-warmup
      LOG_ONCE(info, "[scheduler] Mini-batch size warmup {}", std::string(mbWarmup));
      // This ramps up MB size at start, relative to progress within warm-up period.
      size_t progress = state_->getProgressIn(mbWarmup.unit); // number of updates/labels processed
      auto progressRatio = (double)progress / (double)mbWarmup.n; // where are we relatively within target warm-up period
      // if unit is labels, then account for the fact that our increment itself is not constant
#if 1  // this seems to hurt convergence quite a bit compared to when updates is used
      if (mbWarmup.unit == SchedulingUnit::trgLabels)
        progressRatio = std::sqrt(progressRatio);
#endif
      if (progressRatio < 1)
        ratio *= progressRatio;
    }

    // dynamic MB-size tracking with learning rate
    // As LR goes down, MB gets ramped up by the same ratio, which has been found to be safe.
    auto mbTracking = options_->get<bool>("mini-batch-track-lr");
    if (mbTracking) {
      ABORT("Please review this code");
      auto lrFactor = getScheduledLRDecayFactor(*state_) * state_->factor; // (don't include lr-warmup)
      if (lrFactor != 1)
        LOG_ONCE(info, "[scheduler] Dynamic mini-batch size adjustment enabled and kicking in");
      ratio /= lrFactor;
    }
    return ratio;
  }

  std::tuple<size_t, float, float> getGradientNormStats() const {
    return std::make_tuple(gradientNormAvgWindow_, state_->gradientNormAvg, state_->gradientNormVar);
  }

  std::tuple<size_t, float, float> getLogGradientNormStats() const {
    return std::make_tuple(gradientNormAvgWindow_, state_->logGradientNormAvg, state_->logGradientNormVar);
  }

  bool keepGoing() {
    if(saveAndExitRequested()) // via SIGTERM
      return false;

#if 1  // @TODO: to be removed once we deprecate after-epochs and after-batches
    // stop if it reached the maximum number of epochs
    size_t stopAfterEpochs = options_->get<size_t>("after-epochs");
    if(stopAfterEpochs > 0 && calculateLogicalEpoch() > stopAfterEpochs)
      return false;

    // stop if it reached the maximum number of batch updates
    size_t stopAfterBatches = options_->get<size_t>("after-batches");
    if(stopAfterBatches > 0 && state_->batches >= stopAfterBatches)
      return false;
#endif

    // get list of stopping criteria e.g. "10e,300Ku,20Gt" (10 epochs, 300,000 updates, 20 billion target labels)
    // and stop for whatever criterion hits first.
    std::vector<std::string> stoppingCriteria = utils::split(options_->get<std::string>("after"), ",");
    for(auto stoppingCriterionString : stoppingCriteria) {
      SchedulingParameter stoppingCriterion = SchedulingParameter::parse(stoppingCriterionString);
      if(stoppingCriterion.n > 0) { // is any stopping criterion defined?
        if(stoppingCriterion.unit == SchedulingUnit::epochs    && calculateLogicalEpoch() >  stoppingCriterion.n) return false;
        if(stoppingCriterion.unit == SchedulingUnit::updates   && state_->batches         >= stoppingCriterion.n) return false;
        if(stoppingCriterion.unit == SchedulingUnit::trgLabels && state_->labelsTotal     >= stoppingCriterion.n) return false;
      }
    }

    // stop if the first/all/any validators did not improve for a given number of checks
    size_t stopAfterStalled = options_->get<size_t>("early-stopping");
    if(stopAfterStalled > 0 && stalled() >= stopAfterStalled)
      return false;

    // stop if data streaming from STDIN is stopped
    if(endOfStdin_)
      return false;

    return true;
  }

  void increaseEpoch() {
    LOG(info, "Seen {} samples", utils::withCommas(state_->samplesEpoch));
    state_->newEpoch();
    if(std::to_string(logicalEpoch_) == "1e")
      LOG(info, "Starting epoch {}", state_->epochs);
    else
      LOG(info, "Starting data epoch {} in logical epoch {}", state_->epochs, formatLogicalEpoch());
  }

  void started() { LOG(info, "Training started"); }
  void finished() {
    if (saveAndExitRequested())
      LOG(info, "Training interrupted (via signal).");
    else
      LOG(info, "Training finished");
  }

  void addValidator(Ptr<ValidatorBase> validator) {
    validators_.push_back(validator);

    registerTrainingObserver(validators_.back());
    if(!state_->loaded) {
      state_->validators[validator->type()]["last-best"] = validator->initScore();
      state_->validators[validator->type()]["stalled"] = 0;
    }
    if(validators_.size() == 1)
      state_->validator = validator->type();
  }

  bool validating() {
    return (!validators_.empty()
            && state_->enteredNewPeriodOf(options_->get<std::string>("valid-freq"))
            && state_->largerThan(options_->get<std::string>("valid-from"))
            && keepGoing());
  }

  bool saving() {
    return (state_->enteredNewPeriodOf(options_->get<std::string>("save-freq"))
            && state_->largerThan(options_->get<std::string>("save-from")) );
  }

  bool syncing() {
    return state_->enteredNewPeriodOf(options_->get<std::string>("sync-freq", "0"));
  }

  bool replacingWithSmoothed() {
    if(options_->get<float>("exponential-smoothing", 0.f) != 0.f)
      return state_->enteredNewPeriodOf(options_->get<std::string>("exponential-smoothing-replace-freq", "0"));
    else
      return false;
  }

  void validate(const std::vector<Ptr<ExpressionGraph>>& graphs,
                bool isFinal = false) {
    // Do not validate if already validated (for instance, after the model is loaded)
    // or if validation is scheduled for another update, or when a graceful shutdown
    // was requested.
    if(saveAndExitRequested()
       || state_->validated // already validated (in resumed training, for example)
       || (!state_->enteredNewPeriodOf(options_->get<std::string>("valid-freq")) && !isFinal)) // not now
      return;

    size_t stalledPrev = stalled();
    for(auto validator : validators_) {
      if(!validator)
        continue;

      float value = 0;
      if(!mpi_ || mpi_->isMainProcess()) {
        // We run validation only in the main process, but this is risky with MPI.
        // Validators might modify random state etc., maybe we should run validators
        // everywhere, but not report and not save on the other processes.
        value = validator->validate(graphs, state_);
        if(validator->stalled() > 0) {
          LOG_VALID(info,
                    "Ep. {} : Up. {} : {} : {} : stalled {} times (last best: {})",
                    formatLogicalEpoch(),
                    state_->batches,
                    validator->type(),
                    value,
                    validator->stalled(), validator->lastBest());
        } else {
          LOG_VALID(info,
                    "Ep. {} : Up. {} : {} : {} : new best",
                    formatLogicalEpoch(),
                    state_->batches,
                    validator->type(),
                    value);
        }
      }

      if(mpi_) {
        // collect and broadcast validation result to all processes and bring validator up-to-date
        mpi_->bCast(&value, 1, IMPIWrapper::getDataType(&value));

        // @TODO: add function to validator?
        mpi_->bCast(&validator->stalled(), 1, IMPIWrapper::getDataType(&validator->stalled()));
        mpi_->bCast(&validator->lastBest(), 1, IMPIWrapper::getDataType(&validator->lastBest()));
      }

      state_->validators[validator->type()]["last-best"] = validator->lastBest();
      state_->validators[validator->type()]["stalled"]   = validator->stalled();
    }

    // notify training observers about stalled validation
    size_t stalledNew = stalled();
    if(stalledNew > stalledPrev)
      state_->newStalled(stalledNew);

    state_->validated = true;
  }

  // Returns the proper number of stalled validation w.r.t. early-stopping-on
  size_t stalled() {
    std::string stopOn = options_->get<std::string>("early-stopping-on");
    if(stopOn == "any")
      return stalledMax();
    if(stopOn == "all")
      return stalledMin();
    return stalled1st();
  }

  // Returns the number of stalled validations for the first validator
  size_t stalled1st() {
    if(!validators_.empty())
      if(validators_[0])
        return validators_[0]->stalled();
    return 0;
  }

  // Returns the largest number of stalled validations across validators or 0 if there are no validators
  size_t stalledMax() {
    size_t max = 0;
    for(auto validator : validators_)
      if(validator && validator->stalled() > max)
        max = validator->stalled();
    return max;
  }

  // Returns the lowest number of stalled validations across validators or 0 if there are no validators
  size_t stalledMin() {
    size_t min = std::numeric_limits<std::size_t>::max();
    for(auto validator : validators_)
      if(validator && validator->stalled() < min)
        min = validator->stalled();
    return min == std::numeric_limits<std::size_t>::max() ? 0 : min;
  }

  void update(StaticLoss rationalLoss, Ptr<data::Batch> batch) {
    update(rationalLoss, /*numReadBatches=*/1, /*batchSize=*/batch->size(), /*batchLabels=*/batch->wordsTrg(), /*gradientNorm=*/0.f);
  }

  // @TODO: go back to function which takes batch as an argument? The current arguments make it hard
  // to choose which subbatch should be used for speed display. For sequence-classifiers it's more interesting
  // to see the source-words consumed rather than the labels.
  void update(StaticLoss rationalLoss,
              size_t numReadBatches, // number of batches read by the reader (for seeking in case of restart)
              size_t batchSize,      // total number of sentences in batch
              size_t batchLabels,    // total number of target words in batch
              float gradientNorm) {  // gradientNorm of update
    state_->rememberPreviousProgress();  // note: epoch increases happen at the wrong place, hence
                                         // -freq parameters do not support epoch units
    state_->validated = false;

    // collect costs from all nodes if training with MPI
    if(mpi_) {
      mpi_->allReduce(&rationalLoss.loss,  &rationalLoss.loss,  1, MPI_FLOAT, MPI_SUM);
      mpi_->allReduce(&rationalLoss.count, &rationalLoss.count, 1, MPI_FLOAT, MPI_SUM);
    }
    float currentNormalizedLoss = rationalLoss.loss / rationalLoss.count;

    state_->costSum   += rationalLoss.loss;
    state_->costCount += rationalLoss.count;

    state_->updatesDisp  += 1;
    state_->samplesDisp  += batchSize;
    state_->wordsDisp    += batchLabels; // words at given input processed since last display, for speed display

    state_->samplesEpoch += batchSize;   // sentences processed in this epoch
    state_->labelsTotal  += batchLabels; // total labels processed      

    state_->newUpdate(numReadBatches);

    // true if --throw-on-divergence [lossAvgWindowSlow_] [lossAvgWindowFast_] [divergenceTolerance_] is enabled, false otherwise
    if(throwOnDivergence_ && isFinite(currentNormalizedLoss)) {
      size_t windowSlow = std::min(lossAvgWindowSlow_, state_->batches); // we compare the running exponential average over a longer window
      size_t windowFast = std::min(lossAvgWindowFast_, state_->batches); // with the running exponential everage over a shorter window (for smoothing)
      
      // By default we set windowSlow = 100 and windowFast = 10, so if values diverge the average from the shorter window should pick this up quickly
      // vs the longer window while still smoothing over multiple updates avoiding detecting random single spikes as divergence.
      float alphaSlow = 2.f / (float)(windowSlow + 1); // about 90% of the mass will come from the windowSlow last steps
      float alphaFast = 2.f / (float)(windowFast + 1); // about 90% of the mass will come from the windowFast last steps
  
      // set some reasonable defaults during training start. Cost shouldn't be zero unless fresh start without *.progress.yml
      if(state_->lossAvgSlow == 0) {
        state_->lossAvgSlow = currentNormalizedLoss;
        state_->lossAvgFast = currentNormalizedLoss;
        state_->lossVarSlow = 0;
      }
        
      // allow statistics to see at least lossAvgWindowSlow_ updates before using for divergence detection
      if(state_->batches > lossAvgWindowSlow_) {
        // we compare the faster moving average against the slower moving exponential loss average
        float delta = state_->lossAvgFast - state_->lossAvgSlow;
        // running standard deviation
        float sigma = std::sqrt(state_->lossVarSlow);

        // negative delta is always safe (indicates convergence) and sigma should always be larger than zero (safe for division) after a few first steps
        if(delta > 0 && sigma > 0) {
          // how many standard deviations (sigmas) above slow-moving average?
          float sigmasDiverged = delta / sigma;
          if(sigmasDiverged > divergenceTolerance_) { // uh-oh - by default assume training diverged if slow-moving average is exceeded by e.g. 3 sigmas
            LOG(warn, 
                "Detected training divergence: slow-moving average loss {:.4f} exceeded by fast-moving average loss {:.4f} by {:.4f} = {:.4f} * sigmas", 
                state_->lossAvgSlow, state_->lossAvgFast, delta, sigmasDiverged);

            // this gets propagated to the main training loop in training/training.h and will either fail the whole training process with
            // an unhandled exception (thus exiting with error code) or trigger another training run with fallback to fp32 if we were 
            // training with fp16 and --fp16-fallback-to-fp32 is enabled.
            throw DivergenceException(state_->lossAvgSlow, state_->lossAvgFast, sigmasDiverged);
          }
        }

        if(state_->enteredNewPeriodOf(options_->get<std::string>("disp-freq")) || state_->batches <= options_->get<size_t>("disp-first")) {
          if(!mpi_ || mpi_->isMainProcess()) {
            LOG(debug, 
                "delta(={:.4f}) = avgFast(={:.4f}) - avgSlow(={:.4f}) = {:.4f} * sigma(={:.4f}) < {:.4f} * sigma",
                delta, state_->lossAvgFast, state_->lossAvgSlow, delta / sigma, sigma, divergenceTolerance_);
          }
        }
      }

      // purely diagnostic. This will throw a divergence exception once the specified training progress has occurred. 
      if(throwAfter_) {
        if(state_->enteredNewPeriodOf(throwAfter_)) {
          LOG(warn, "Training reached {}; throwing diagnostic DivergenceException", (std::string)throwAfter_);
          throw DivergenceException(state_->lossAvgSlow, state_->lossAvgFast, 0.f);
        }
      }
      
      // log slow-moving exponential average and variance of training cost stats
      float deltaSlow = currentNormalizedLoss - state_->lossAvgSlow;
      state_->lossAvgSlow = state_->lossAvgSlow + alphaSlow * deltaSlow;
      state_->lossVarSlow  = (1.0f - alphaSlow) * (state_->lossVarSlow + alphaSlow * deltaSlow * deltaSlow);
      
      // log fast-moving exponential average of training cost stats
      float deltaFast = currentNormalizedLoss - state_->lossAvgFast;
      state_->lossAvgFast = state_->lossAvgFast + alphaFast * deltaFast;
    }

    if(gradientNorm) {
      size_t range = std::min(gradientNormAvgWindow_, state_->batches);
      float alpha = 2.f / (float)(range + 1);

      float delta = gradientNorm - state_->gradientNormAvg;
      state_->gradientNormAvg = state_->gradientNormAvg + alpha * delta;
      state_->gradientNormVar = (1.0f - alpha) * (state_->gradientNormVar + alpha * delta * delta);

      float logDelta = std::log(gradientNorm) - state_->logGradientNormAvg;
      state_->logGradientNormAvg = state_->logGradientNormAvg + alpha * logDelta;
      state_->logGradientNormVar = (1.0f - alpha) * (state_->logGradientNormVar + alpha * logDelta * logDelta);
    }

    // reconstruct sum cost, for displaying epoch-level averages instead of minibatch-level
    auto lossType = options_->get<std::string>("cost-type");
    auto dispLabelCounts = options_->get<bool>("disp-label-counts");  // if true then show as "cost per label * number of labels"

    if(state_->enteredNewPeriodOf(options_->get<std::string>("disp-freq")) || state_->batches <= options_->get<size_t>("disp-first")) {
      // if MPI then aggregate precise cost across workers
      if(!mpi_ || mpi_->isMainProcess()) {
        if(options_->get<bool>("lr-report")) {
          LOG(info,
              "Ep. {} : Up. {} : Sen. {} : {} : Time {:.2f}s : {:.2f} words/s : gNorm {:.4f} : L.r. {:.4e}",
              formatLogicalEpoch(),
              state_->batches,
              utils::withCommas(state_->samplesEpoch),
              formatLoss(lossType, dispLabelCounts, batchLabels, state_),
              timer_.elapsed(),
              state_->wordsDisp / timer_.elapsed(),
              state_->gradientNormAvg,
              state_->eta);
        } else {
          LOG(info,
              "Ep. {} : Up. {} : Sen. {} : {} : Time {:.2f}s : {:.2f} words/s : gNorm {:.4f}",
              formatLogicalEpoch(),
              state_->batches,
              utils::withCommas(state_->samplesEpoch),
              formatLoss(lossType, dispLabelCounts, batchLabels, state_),
              timer_.elapsed(),
              state_->wordsDisp / timer_.elapsed(),
              state_->gradientNormAvg);
        }
      }
      timer_.start();
      state_->costSum      = 0;
      state_->costCount    = 0;

      state_->updatesDisp  = 0;
      state_->samplesDisp  = 0;
      state_->wordsDisp    = 0;
    }
  }

  void loadFromString(const std::string yamlString) {
    if(!yamlString.empty())
      state_->loadFromString(yamlString);

    if(options_->get<bool>("no-restore-corpus")) {
      state_->samplesEpoch = 0;
      state_->costSum      = 0;
      state_->costCount    = 0;

      state_->updatesDisp  = 0;
      state_->samplesDisp  = 0;
      state_->wordsDisp    = 0;
    }

    if(options_->get<bool>("valid-reset-stalled") || options_->get<bool>("valid-reset-all")) {
      state_->stalled      = 0;
      state_->maxStalled   = 0;
      for(const auto& validator : validators_) {
        if(state_->validators[validator->type()]) {
          // reset the number of stalled validations, e.g. when the validation set is the same
          state_->validators[validator->type()]["stalled"] = 0;
          // reset last best results as well, e.g. when the validation set changes
          if(options_->get<bool>("valid-reset-all"))
            state_->validators[validator->type()]["last-best"] = validator->initScore();
        }
      }
    }

    state_->newLoad();
  }

  void load(const std::string& name) {
    std::string nameYaml = name + ".progress.yml";
    std::string yamlStr;
    if(mpi_->isMainProcess())
      if(filesystem::exists(nameYaml))
        yamlStr = io::InputFileStream(nameYaml).readToString();

    if(mpi_)
      mpi_->bCast(yamlStr);

    loadFromString(yamlStr);
  }

  void save(const std::string& name) {
    // Save config options
    std::ofstream fout(name + ".yml");
    fout << options_->asYamlString();
    // Save training progress
    state_->save(name + ".progress.yml");
  }

  size_t numberOfBatches() { return state_->batches; }

  void registerTrainingObserver(Ptr<TrainingObserver> observer) {
    state_->registerObserver(observer);
  }

  void actAfterEpoch(TrainingState& state) override {
    // stop if data streaming from STDIN is stopped for a TSV input
    std::string firstPath = options_->get<std::vector<std::string>>("train-sets")[0];
    if(options_->get<bool>("tsv", false) && (firstPath == "stdin" || firstPath == "-"))
      endOfStdin_ = true;

    float factor = options_->get<float>("lr-decay");

    updateLearningRate(state);

    if(factor > 0.0) {
      bool decay = false;
      auto strategy = options_->get<std::string>("lr-decay-strategy");
      state.reset = false;

      if(strategy == "epoch" || strategy == "epoch+batches"
         || strategy == "epoch+stalled") {
        size_t startEpoch
            = options_->get<std::vector<size_t>>("lr-decay-start").front();
        if(startEpoch && state.epochs >= startEpoch)
          decay = true;
      }

      if(strategy == "epoch+batches") {
        size_t startBatches
            = options_->get<std::vector<size_t>>("lr-decay-start")[1];
        if(startBatches && state.batches >= startBatches)
          decay = true;
      }
      if(strategy == "epoch+stalled") {
        size_t startStalled
            = options_->get<std::vector<size_t>>("lr-decay-start")[1];
        if(startStalled && state.maxStalled >= startStalled)
          decay = true;
      }

      if(decay) {
        state.factor *= factor;
        updateLearningRate(state);
        LOG(info, "Decaying learning rate to {} in epoch {}", state.eta, state.epochs);

        state.reset = options_->get<bool>("lr-decay-reset-optimizer");
        if(state.reset)
          LOG(info, "Resetting optimizer statistics");

        if(options_->get<bool>("lr-decay-repeat-warmup")) {
          LOG(info, "Restarting learning rate warmup");
          state.warmupStart.n = state.getProgressIn(SchedulingParameter::parse(options_->get<std::string>("lr-warmup")).unit);
        }
      }
    }
  }

  void actAfterBatches(TrainingState& state) override {
    float factor = options_->get<float>("lr-decay");
    state.reset = false;

    updateLearningRate(state);

    if(factor > 0.0) {
      if(options_->get<std::string>("lr-decay-strategy") == "batches") {
        size_t start = options_->get<std::vector<size_t>>("lr-decay-start").front();
        size_t freq  = options_->get<size_t>("lr-decay-freq"); // note: unlike e.g. disp-freq, this is always in batches

        if(start > 0 && freq > 0 && state.batches >= start
           && ((state.batches - start) % freq == 0)) {
          state.factor *= factor;
          updateLearningRate(state);
          LOG(info, "Decaying learning rate to {} after {} batches", state.eta, state.batches);

          state.reset = options_->get<bool>("lr-decay-reset-optimizer");
          if(state.reset)
            LOG(info, "Resetting optimizer statistics");

          if(options_->get<bool>("lr-decay-repeat-warmup")) {
            LOG(info, "Restarting learning rate warmup");
            // TODO: avoid repeating this many times and minimize calls to options_->get
            state.warmupStart.n = state.getProgressIn(SchedulingParameter::parse(options_->get<std::string>("lr-warmup")).unit);
          }
        }
      }
    }

    if(first_ && options_->get<bool>("lr-warmup-at-reload")) {
      LOG(info, "Restarting learning rate warmup");
      state.warmupStart.n = state.getProgressIn(SchedulingParameter::parse(options_->get<std::string>("lr-warmup")).unit);
    }

    if(options_->get<bool>("lr-warmup-cycle")) {
      if(state_->enteredNewPeriodOf(options_->get<std::string>("lr-warmup")))
        state.warmupStart.n = state.getProgressIn(SchedulingParameter::parse(options_->get<std::string>("lr-warmup")).unit);
    }

    first_ = false;
  }

  void actAfterStalled(TrainingState& state) override {
    float factor = options_->get<float>("lr-decay");
    state.reset = false;

    updateLearningRate(state);

    if(factor > 0.0) {
      if(options_->get<std::string>("lr-decay-strategy") == "stalled") {
        size_t startStalled = options_->get<std::vector<size_t>>("lr-decay-start").front();
        if(startStalled && state.stalled && state.stalled % startStalled == 0) {
          state.factor *= factor;
          updateLearningRate(state);
          LOG(info,
              "Decaying learning rate to {} after having stalled {} time(s)",
              state.eta,
              state.stalled);

          state.reset = options_->get<bool>("lr-decay-reset-optimizer");
          if(state.reset)
            LOG(info, "Resetting optimizer statistics");

          if(options_->get<bool>("lr-decay-repeat-warmup")) {
            LOG(info, "Restarting learning rate warmup");
            state.warmupStart.n = state.getProgressIn(
                SchedulingParameter::parse(options_->get<std::string>("lr-warmup")).unit);
          }
        }
      }
    }
  }
};
}  // namespace marian
