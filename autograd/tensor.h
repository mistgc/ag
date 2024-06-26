#ifndef _AUTOGRAD_TENSOR_H_
#define _AUTOGRAD_TENSOR_H_

#include <cassert>
#include <cstddef>
#include <optional>
#include <vector>

namespace autograd {

struct Tensor;

struct Dependency {};

struct Tensor {
  using Dependencise = std::vector<Dependency>;
  using Shape = std::vector<size_t>;
  using Index = std::vector<size_t>;

  std::vector<float> data;
  std::vector<size_t> shape;
  bool isRequiredGrad;
  Tensor *grad{nullptr};
  Dependencise deps;

  Tensor(bool requiresGrad = false) : shape({1}), isRequiredGrad(requiresGrad) {
    this->grad = new Tensor();
  }

  Tensor(Shape &shape, bool requiresGrad = false)
      : shape(shape), isRequiredGrad(requiresGrad) {
    this->grad = new Tensor();
  }

  ~Tensor() { delete this->grad; }

  bool checkIndex(Index &index) {
    size_t size = index.size();
    bool flag = size == this->shape.size();

    for (size_t i = 0; i < size; i++) {
      if (index[i] >= this->shape[i] || index[i] < 0) {
        flag = false;
        break;
      }
    }

    return flag;
  }

  bool checkIndex(Index &&index) {
    size_t size = index.size();
    bool flag = size == this->shape.size();

    for (size_t i = 0; i < size; i++) {
      if (index[i] >= this->shape[i] || index[i] < 0) {
        flag = false;
        break;
      }
    }

    return flag;
  }

  size_t calcRawIndex(Index &index) {
    assert(checkIndex(index) && "The passed index is invalid.");
    size_t rawIndex = 0;
    size_t indexSize = index.size();

    for (size_t i = 0; i < indexSize - 1; i++) {
      size_t tmp = 1;
      for (size_t j = i + 1; j < indexSize - 1; j++) {
        tmp *= this->shape[j];
      }
      rawIndex = tmp * i;
    }
    rawIndex += index[indexSize - 1];
    return rawIndex;
  }

  size_t calcRawIndex(Index &&index) {
    assert(checkIndex(index) && "The passed index is invalid.");
    size_t rawIndex = 0;
    size_t indexSize = index.size();

    for (size_t i = 0; i < indexSize - 1; i++) {
      size_t tmp = 1;
      for (size_t j = i + 1; j < indexSize - 1; j++) {
        tmp *= this->shape[j];
      }
      rawIndex = tmp * i;
    }
    rawIndex += index[indexSize - 1];
    return rawIndex;
  }

  inline float at(Index &index) { return this->data[calcRawIndex(index)]; }

  inline float at(Index &&index) { return this->data[calcRawIndex(index)]; }

  inline void set(Index &index, float value) {
    this->data[calcRawIndex(index)] = value;
  }

  inline void set(Index &&index, float value) {
    this->data[calcRawIndex(index)] = value;
  }

  inline float get(Index &index) { return at(index); }

  inline float get(Index &&index) { return at(index); }

  Tensor backward(std::optional<Tensor> grad_ = std::nullopt) {}

  bool isScaleValue() { return this->shape.size() == 1 && this->shape[0] == 1; }
};

} // namespace autograd

#endif // _AUTOGRAD_TENSOR_H_
