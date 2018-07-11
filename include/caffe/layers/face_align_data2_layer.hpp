#ifndef CAFFE_FACE_ALIGN_DATA2_LAYER_HPP_
#define CAFFE_FACE_ALIGN_DATA2_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/face_align_base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {

template <typename Dtype>
class FaceAlignData2Layer : public FaceAlignBasePrefetchingDataLayer<Dtype> {
 public:
  explicit FaceAlignData2Layer(const LayerParameter& param);
  virtual ~FaceAlignData2Layer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "FaceAlignData2"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

  cv::Mat bestFitRect(const cv::Mat& groundTruth, const cv::Mat& meanShape);
  cv::Mat mirrorShape(const cv::Mat& groundTruth, const cv::Mat& image);
  bool generatePerturbation(
      const cv::Mat& image,
      const cv::Mat& groundTruth,
      const cv::Rect_<float>& face_box,
      cv::Mat& tempImg,
      cv::Mat& tempGroundTruth);
  
  void bestFitMat(const cv::Mat& destination, const cv::Mat& source, cv::Mat& A, cv::Mat& t);

  void boundingRect(const cv::Mat& shape, cv::Rect_<float>& rect);
  void meanXYOfShape(const cv::Mat& shape, float& x, float &y);

 protected:
  void Next();
  bool Skip();
  virtual void load_batch(FaceAlignBatch<Dtype>* batch);

  shared_ptr<db::DB> db_;
  shared_ptr<db::Cursor> cursor_;
  uint64_t offset_;
  
  int num_landmark_;
  cv::Mat mean_shape_;
  int new_image_size_;
  float frame_fraction_;

  bool random_mirror_;
  float mirror_prob_;

  float translationX_prob_;
  bool random_translationX_;
  float translationMultX_;
  float translationY_prob_;
  bool random_translationY_;
  float translationMultY_;

  float rotation_prob_;
  bool random_rotation_;
  float rotationStdDev_;

  float scale_prob_;
  bool random_scale_;
  float scaleStdDev_;

  bool visualation_;
  int visualation_step_;
};

}  // namespace caffe

#endif  // CAFFE_FACE_ALIGN_DATA2_LAYER_HPP_
