#pragma once

#include "marian.h"

#include "data/shortlist.h"
#include "layers/factory.h"

namespace marian {
namespace mlp {
/**
 * @brief Activation functions
 */
enum struct act : int { linear, tanh, sigmoid, ReLU, LeakyReLU, PReLU, swish };
}  // namespace mlp
}  // namespace marian

namespace marian {

// Each layer consists of LayerBase and IXXXLayer which defines one or more apply()
// functions for the respective layer type (different layers may require different signatures).
// This base class contains configuration info for creating parameters and executing apply().
class LayerBase {
protected:
  Ptr<ExpressionGraph> graph_;
  Ptr<Options> options_;

public:
  LayerBase(Ptr<ExpressionGraph> graph, Ptr<Options> options) : graph_(graph), options_(options) {}

  template <typename T>
  T opt(const std::string key) const {
    return options_->get<T>(key);
  }

  template <typename T>
  T opt(const std::string key, const T& defaultValue) const {
    return options_->get<T>(key, defaultValue);
  }
};

// Simplest layer interface: Unary function
struct IUnaryLayer {
  virtual ~IUnaryLayer() {}
  virtual Expr apply(Expr) = 0;
  virtual Expr apply(const std::vector<Expr>& es) {
    ABORT_IF(es.size() > 1, "Not implemented");  // simple stub
    return apply(es.front());
  }
};

struct IHasShortList {
  virtual void setShortlist(Ptr<data::Shortlist> shortlist) = 0;
  virtual void clear() = 0;
};

// Embedding from corpus sub-batch to (emb, mask)
struct IEmbeddingLayer {
  virtual std::tuple<Expr /*embeddings*/, Expr /*mask*/> apply(
      Ptr<data::SubBatch> subBatch) const = 0;

  virtual Expr apply(const Words& embIdx, const Shape& shape) const = 0;

  // alternative from indices directly
  virtual Expr applyIndices(const std::vector<WordIndex>& embIdx, const Shape& shape) const = 0;
  virtual ~IEmbeddingLayer() {}
};

// base class for Encoder and Decoder classes, which have embeddings and a batch index (=stream
// index)
class EncoderDecoderLayerBase : public LayerBase {
protected:
  const std::string prefix_;
  const bool embeddingFix_;
  const float dropoutEmbeddings_;  // this drops out full embedding vectors
  const bool inference_;
  const size_t batchIndex_;
  mutable std::vector<Ptr<IEmbeddingLayer>> embeddingLayers_;  // (lazily created)

  EncoderDecoderLayerBase(Ptr<ExpressionGraph> graph,
                          Ptr<Options> options,
                          const std::string& prefix,
                          size_t batchIndex,
                          float dropoutEmbeddings,
                          bool embeddingFix)
      : LayerBase(graph, options),
        prefix_(options->get<std::string>("prefix", prefix)),
        embeddingFix_(embeddingFix),
        dropoutEmbeddings_(dropoutEmbeddings),
        inference_(options->get<bool>("inference", false)),
        batchIndex_(options->get<size_t>("index", batchIndex)) {}

  virtual ~EncoderDecoderLayerBase() {}

private:
  Ptr<IEmbeddingLayer> createEmbeddingLayer() const;
  Ptr<IEmbeddingLayer> createULREmbeddingLayer() const;

public:
  // get embedding layer; lazily create on first call
  Ptr<IEmbeddingLayer> getEmbeddingLayer(bool ulr = false) const;
};

namespace mlp {

class Dense : public LayerBase, public IUnaryLayer {
public:
  Dense(Ptr<ExpressionGraph> graph, Ptr<Options> options) : LayerBase(graph, options) {}

  Expr apply(const std::vector<Expr>& inputs) override {
    ABORT_IF(inputs.empty(), "No inputs");

    auto name = opt<std::string>("prefix");
    auto dim = opt<int>("dim");

    auto useLayerNorm = opt<bool>("layer-normalization", false);
    auto useNematusNorm = opt<bool>("nematus-normalization", false);
    auto activation = (act)opt<int>("activation", (int)act::linear);

    auto g = graph_;

    std::vector<Expr> outputs;
    size_t i = 0;

    std::string num;
    for(auto&& in : inputs) {
      if(inputs.size() > 1)
        num = std::to_string(i);

      Expr W = g->param(name + "_W" + num, {in->shape()[-1], dim}, inits::glorotUniform());
      Expr b = g->param(name + "_b" + num, {1, dim}, inits::zeros());

      if(useLayerNorm) {
        if(useNematusNorm) {
          auto ln_s = g->param(name + "_ln_s" + num, {1, dim}, inits::fromValue(1.f));
          auto ln_b = g->param(name + "_ln_b" + num, {1, dim}, inits::zeros());

          outputs.push_back(layerNorm(affine(in, W, b), ln_s, ln_b, NEMATUS_LN_EPS));
        } else {
          auto gamma = g->param(name + "_gamma" + num, {1, dim}, inits::fromValue(1.0));

          outputs.push_back(layerNorm(dot(in, W), gamma, b));
        }
      } else {
        outputs.push_back(affine(in, W, b));
      }
      i++;
    }

    // clang-format off
    switch(activation) {
      case act::linear:    return plus(outputs);
      case act::tanh:      return tanh(outputs);
      case act::sigmoid:   return sigmoid(outputs);
      case act::ReLU:      return relu(outputs);
      case act::LeakyReLU: return leakyrelu(outputs);
      case act::PReLU:     return prelu(outputs);
      case act::swish:     return swish(outputs);
      default:             return plus(outputs);
    }
    // clang-format on
  };

  Expr apply(Expr input) override { return apply(std::vector<Expr>({input})); }
};

}  // namespace mlp

// --- a few layers with built-in parameters created on the fly, without proper object
// @TODO: change to a proper layer object

// like affine() but with built-in parameters, activation, and dropout
static inline Expr denseInline(Expr x,
                               std::string prefix,
                               std::string suffix,
                               int outDim,
                               Ptr<inits::NodeInitializer> initFn = inits::glorotUniform(),
                               const std::function<Expr(Expr)>& actFn = nullptr,
                               float dropProb = 0.0f) {
  auto graph = x->graph();

  auto W = graph->param(prefix + "_W" + suffix, {x->shape()[-1], outDim}, inits::glorotUniform());
  auto b = graph->param(prefix + "_b" + suffix, {1, outDim}, inits::zeros());

  x = affine(x, W, b);
  if(actFn)
    x = actFn(x);
  x = dropout(x, dropProb);  // @TODO: check for infernce?
  return x;
}

static inline Expr layerNorm(Expr x, std::string prefix, std::string suffix = std::string()) {
  int dimModel = x->shape()[-1];
  auto scale = x->graph()->param(prefix + "_ln_scale" + suffix, {1, dimModel}, inits::ones());
  auto bias = x->graph()->param(prefix + "_ln_bias" + suffix, {1, dimModel}, inits::zeros());
  return marian::layerNorm(x, scale, bias, 1e-6f);
}

}  // namespace marian
