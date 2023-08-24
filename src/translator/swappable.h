#pragma once
/* Support for swapping models in and out of a GPU, when you have more models
 * than fit in the GPU's RAM.  The models must have identical graphs, including
 * size. They can have different parameters and different vocabularies but the
 * vocabularies must have the same size.  To make vocabulary the same size, pad
 * using scripts/contrib/pad_model_vocabulary.py offline.
 */
#include "marian.h"
#include "common/io.h"
#include "data/vocab.h"
#include "translator/history.h"

#include <string>
#include <vector>
namespace marian {

class Scorer;

class GPULoadedModel;
class CPULoadedModel;

/* Execute on a particular device */
class GPUEngine {
	private:
    friend class GPULoadedModel;
    Ptr<Options> options_;
    Ptr<ExpressionGraph> graph_;
    std::vector<Ptr<Scorer> > scorers_;
    const DeviceId myDeviceId_;
    Allocator allocator_;

    void SwapPointers(std::vector<MemoryPiece::PtrType> &with);

  public:
    /**
     * @param options The marian options object
     * @param deviceNum The index of the device you want to use for this slot. Note that this is not the deviceID but the index of the device in the
     *                  array of supplied devices. Eg if you provide -d 0 3 5 and you want the Slot to run on GPU 3, you provide deviceNum=1.
     */
    explicit GPUEngine(Ptr<Options> options, size_t deviceNum);

    ~GPUEngine();
};

/* A model loaded on the GPU that can be overwritten from CPU or GPU. */
class GPULoadedModel {
  private:
    Ptr<GPUEngine> engine_;

    std::vector<MemoryPiece::PtrType> parameters_;
    std::vector<Ptr<Vocab>> srcVocabs_;
    Ptr<Vocab> trgVocab_;

  public:
    GPULoadedModel(Ptr<GPUEngine> gpu);

    ~GPULoadedModel();

    const std::vector<Ptr<Vocab>> &SrcVocabs() const { return srcVocabs_; }

    Ptr<Vocab> TrgVocab() const { return trgVocab_; }

    // Overwrite this model with parameters from a different one.
    void Load(const CPULoadedModel &from);
    void Load(const GPULoadedModel &from);

    Histories Translate(const std::vector<std::string> &input, float max_length_factor = -1.f);
};

/* A model loaded on the CPU. */
class CPULoadedModel {
  private:
    std::vector<io::Item> parameters_;
    std::vector<Ptr<Vocab>> srcVocabs_;
    Ptr<Vocab> trgVocab_;

  public:
    // The parts of Options that relate to model and vocab are ignored.  The files provided will be loaded.
    CPULoadedModel(Ptr<Options> options, const std::string &parameters, const std::vector<std::string> &sourceVocabPaths, const std::string &targetVocabPath)
            : CPULoadedModel(std::move(options), io::loadItems(parameters), sourceVocabPaths, targetVocabPath) {};

    CPULoadedModel(Ptr<Options> options, std::vector<io::Item> parameters, const std::vector<std::string> &sourceVocabPaths, const std::string &targetVocabPath);

    void CopyTo(Ptr<ExpressionGraph> graph) const;

    const std::vector<io::Item> &Parameters() const { return parameters_; }

    const std::vector<Ptr<Vocab>> &SrcVocabs() const { return srcVocabs_; }

    Ptr<Vocab> TrgVocab() const { return trgVocab_; }
};

class GPUSlot {
public:
    const DeviceId device;

    explicit GPUSlot(const Ptr<ExpressionGraph> &graph);

    ~GPUSlot();

    inline std::vector<MemoryPiece::PtrType> &params() {
        return parameters_;
    }

private:
    Allocator allocator_;
    std::vector<MemoryPiece::PtrType> parameters_;
};

} // namespace marian
