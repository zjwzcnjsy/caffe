#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#endif // USE_OPENCV
#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/face_align_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe
{

template <typename Dtype>
void bestFit(const int num_landmark, const Dtype *dest, const Dtype *src, Dtype *transformed_src, Dtype *T);

template <typename Dtype>
FaceAlignDataLayer<Dtype>::FaceAlignDataLayer(const LayerParameter &param)
    : FaceAlignBasePrefetchingDataLayer<Dtype>(param),
      offset_()
{
  db_.reset(db::GetDB(param.data_param().backend()));
  db_->Open(param.data_param().source(), db::READ);
  cursor_.reset(db_->NewCursor());

  num_landmark_ = param.face_align_data_param().num_landmark();
  new_image_size_ = param.face_align_data_param().new_image_size();
  frame_fraction_ = param.face_align_data_param().frame_fraction();

  random_mirror_ = param.face_align_data_param().random_mirror();
  mirror_prob_ = param.face_align_data_param().mirror_prob();
  translationX_prob_ = param.face_align_data_param().translationx_prob();
  random_translationX_ = param.face_align_data_param().random_translationx();
  translationMultX_ = param.face_align_data_param().translationmultx();
  translationY_prob_ = param.face_align_data_param().translationy_prob();
  random_translationY_ = param.face_align_data_param().random_translationy();
  translationMultY_ = param.face_align_data_param().translationmulty();
  rotation_prob_ = param.face_align_data_param().rotation_prob();
  random_rotation_ = param.face_align_data_param().random_rotation();
  rotationStdDev_ = param.face_align_data_param().rotationstddev();
  scale_prob_ = param.face_align_data_param().scale_prob();
  random_scale_ = param.face_align_data_param().random_scale();
  scaleStdDev_ = param.face_align_data_param().scalestddev();

  visualation_ = param.face_align_data_param().visualation();
  visualation_step_ = param.face_align_data_param().visualation_step();

  max_trials_ = param.face_align_data_param().max_trials();
  min_jaccard_overlap_ = param.face_align_data_param().min_jaccard_overlap();

  const string &mean_shape_file = param.face_align_data_param().mean_shape_file();
  if (Caffe::root_solver())
  {
    LOG(INFO) << "Loading mean shape file from: " << mean_shape_file;
  }
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_shape_file.c_str(), &blob_proto);
  Blob<float> mean_shape;
  mean_shape.FromProto(blob_proto);
  CHECK_EQ(mean_shape.num(), num_landmark_);
  CHECK_EQ(mean_shape.channels(), 2);
  mean_shape_.create(num_landmark_, 2, CV_32FC1);
  const float *data = mean_shape.cpu_data();
  for (int i = 0; i < num_landmark_; ++i)
  {
    for (int j = 0; j < 2; ++j)
    {
      mean_shape_.at<float>(i, j) = data[i * 2 + j];
    }
  }
}

template <typename Dtype>
FaceAlignDataLayer<Dtype>::~FaceAlignDataLayer()
{
  this->StopInternalThread();
}

template <typename Dtype>
void FaceAlignDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                               const vector<Blob<Dtype> *> &top)
{
  const int batch_size = this->layer_param_.data_param().batch_size();
  // Read a data point, and use it to initialize the top blob.
  FaceAlignDatum datum;
  datum.ParseFromString(cursor_->value());
  CHECK_EQ(datum.has_data(), true);
  CHECK_EQ(datum.label_size(), 2 * num_landmark_);

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape(4);
  top_shape[0] = 1;
  top_shape[1] = datum.channels();
  top_shape[2] = new_image_size_;
  top_shape[3] = new_image_size_;
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i)
  {
    this->prefetch_[i]->data_.Reshape(top_shape);
  }
  LOG_IF(INFO, Caffe::root_solver())
      << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_)
  {
    vector<int> label_shape(2);
    label_shape[0] = batch_size;
    label_shape[1] = 2 * num_landmark_;
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->prefetch_.size(); ++i)
    {
      this->prefetch_[i]->label_.Reshape(label_shape);
    }
    LOG_IF(INFO, Caffe::root_solver())
        << "output label size: " << top[1]->num() << ","
        << top[1]->channels();
  }
  if (top.size() >= 3)
  {
    vector<int> pose_shape(2);
    pose_shape[0] = batch_size;
    pose_shape[1] = 3;
    top[2]->Reshape(pose_shape);
    for (int i = 0; i < this->prefetch_.size(); ++i)
    {
      this->prefetch_[i]->pose_.Reshape(pose_shape);
    }
    LOG_IF(INFO, Caffe::root_solver())
        << "output label size: " << top[2]->num() << ","
        << top[2]->channels();
  }
}

template <typename Dtype>
bool FaceAlignDataLayer<Dtype>::Skip()
{
  int size = Caffe::solver_count();
  int rank = Caffe::solver_rank();
  bool keep = (offset_ % size) == rank ||
              // In test mode, only rank 0 runs, so avoid skipping
              this->layer_param_.phase() == TEST;
  return !keep;
}

template <typename Dtype>
void FaceAlignDataLayer<Dtype>::Next()
{
  cursor_->Next();
  if (!cursor_->valid())
  {
    LOG_IF(INFO, Caffe::root_solver())
        << "Restarting data prefetching from start.";
    cursor_->SeekToFirst();
  }
  offset_++;
}

// This function is called on prefetch thread
template <typename Dtype>
void FaceAlignDataLayer<Dtype>::load_batch(FaceAlignBatch<Dtype> *batch)
{
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(batch->label_.count());
  CHECK(this->transformed_data_.count());
  const int batch_size = this->layer_param_.data_param().batch_size();

  FaceAlignDatum datum;
  cv::Mat cur_shape(num_landmark_, 2, CV_32FC1);
  float cur_yaw = 0., cur_pitch = 0., cur_row = 0.;
  vector<cv::Mat> batchSampleImages(batch_size);
  vector<cv::Mat> batchSampleShapes(batch_size);
  vector<float> batchYaw(batch_size);
  vector<float> batchPitch(batch_size);
  vector<float> batchRow(batch_size);
  vector<float> batchAngle(batch_size);
  vector<int> batchMirrorFlag(batch_size);

  for (int item_id = 0; item_id < batch_size; ++item_id)
  {
    timer.Start();
    while (Skip())
    {
      Next();
    }
    datum.ParseFromString(cursor_->value());
    read_time += timer.MicroSeconds();

    cv::Mat image = FaceAlignDatumToCVMat(datum);
    CHECK_EQ(datum.label_size(), 2 * num_landmark_);
    for (int i = 0; i < num_landmark_; ++i)
    {
      for (int j = 0; j < 2; ++j)
      {
        cur_shape.at<float>(i, j) = datum.label(i * 2 + j);
      }
    }
    if (batch->has_pose_)
    {
      cur_yaw = datum.yaw();
      cur_pitch = datum.pitch();
      cur_row = datum.row();
    }
    // for (int i = 0; i < num_landmark_; ++i) {
    //   cv::circle(image, cv::Point(cur_shape.at<float>(i, 0), cur_shape.at<float>(i, 1)),
    //     2, cv::Scalar(0, 0, 255), 2);
    // }
    // Apply data transformations (mirror, scale, crop...)
    timer.Start();

    cv::Mat initLandmark = bestFitRect(cur_shape, mean_shape_);
    float angle = 0.;
    cv::Mat tempInit;
    cv::Mat tempImg, tempInit2, tempGroundTruth;
    int trials = 0;
    bool flag = true;
    do
    {
      tempInit = generatePerturbation(cur_shape, initLandmark, mean_shape_, image, angle);
      
      flag = cropResizeRotate(
          mean_shape_,
          image,
          tempInit,
          cur_shape,
          tempImg,
          tempInit2,
          tempGroundTruth);
      ++trials;
      if (trials >= max_trials_)
      {
        break;
      }
      if (flag)
      {
        break;
      }
    } while (true);

    if (!flag || trials >= max_trials_)
    {
      item_id--;
      Next();
      continue;
    }

    if (batch->has_pose_)
    {
      cur_row += angle;
      batchAngle[item_id] = angle;
    }

    // random mirror
    if (random_mirror_)
    {
      float mirror_prob;
      caffe_rng_uniform<float>(1, 0.f, 1.f, &mirror_prob);
      if (mirror_prob > mirror_prob_)
      {
        cv::Mat tempShape = mirrorShape(tempGroundTruth, tempImg);
        cv::flip(tempImg, tempImg, 1);
        tempGroundTruth = tempShape;
        if (batch->has_pose_)
        {
          cur_yaw = -cur_yaw;
          cur_row = -cur_row;
        }
        batchMirrorFlag[item_id] = 1;
      }
      else
      {
        batchMirrorFlag[item_id] = 0;
      }
    }

    this->data_transformer_->DistortImage(tempImg, tempImg);

    // for (int i = 0; i < tempInit.rows; ++i) {
    //   float x = tempInit.at<float>(i, 0);
    //   float y = tempInit.at<float>(i, 1);
    //   cv::circle(image, cv::Point(x, y), 1, cv::Scalar(255, 0, 0), 1);
    //   x = cur_shape.at<float>(i, 0);
    //   y = cur_shape.at<float>(i, 1);
    //   cv::circle(image, cv::Point(x, y), 1, cv::Scalar(255, 255, 255), 1);
    // }
    // cv::imshow(cv::format("img#%d", item_id), image);

    batchSampleImages[item_id] = tempImg.clone();
    batchSampleShapes[item_id] = tempGroundTruth.clone();

    int offset = batch->data_.offset(item_id);
    Dtype *top_data = batch->data_.mutable_cpu_data();
    this->transformed_data_.set_cpu_data(top_data + offset);
    this->data_transformer_->Transform(tempImg, &(this->transformed_data_));
    // Copy label.
    if (this->output_labels_)
    {
      Dtype *top_label = batch->label_.mutable_cpu_data();
      //caffe_copy(datum.label_size(), datum.label(), top_label);
      for (int i = 0; i < num_landmark_; ++i)
      {
        for (int j = 0; j < 2; ++j)
        {
          top_label[item_id * num_landmark_ * 2 + 2 * i + j] = tempGroundTruth.at<float>(i, j);
        }
      }
    }
    if (batch->has_pose_)
    {
      Dtype *top_pose = batch->pose_.mutable_cpu_data() + item_id * batch->pose_.channels();
      top_pose[0] = cur_yaw / 90.;
      top_pose[1] = cur_pitch / 90.;
      top_pose[2] = cur_row / 90.;
      batchYaw[item_id] = cur_yaw;
      batchPitch[item_id] = cur_pitch;
      batchRow[item_id] = cur_row;
    }
    trans_time += timer.MicroSeconds();
    Next();
  }
  if (visualation_)
  {
    for (int i = 0; i < batchSampleImages.size(); ++i)
    {
      cv::Mat image = batchSampleImages[i];
      cv::Mat shape = batchSampleShapes[i];
      for (int j = 0; j < num_landmark_; ++j)
      {
        cv::circle(image, cv::Point(shape.at<float>(j, 0), shape.at<float>(j, 1)), 1, cv::Scalar(0, 255, 0), 1);
      }
      cv::putText(image, cv::format("#%d#%.2f#%.2f#%.2f$%.2f", batchMirrorFlag[i], batchYaw[i], batchPitch[i], batchRow[i], batchAngle[i]),
                  cv::Point(0, 30), cv::FONT_HERSHEY_SIMPLEX,
                  0.5, cv::Scalar(0, 0, 255), 1);
      cv::imshow(cv::format("image@%d", i), image);
    }
    cv::waitKey(visualation_step_);
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

template <typename Dtype>
cv::Mat FaceAlignDataLayer<Dtype>::generatePerturbation(const cv::Mat &groundTruth,
                                                        const cv::Mat &initLandmark,
                                                        const cv::Mat &meanShape,
                                                        const cv::Mat &image,
                                                        float &angle)
{
  // cv::Rect_<float> boxRect;
  // boundingRect(meanShape, boxRect);
  // float meanShapeSize = std::max<float>(boxRect.height, boxRect.width);
  // float destShapeSize = new_image_size_ * (1 - 2. * frame_fraction_);

  // float scale = destShapeSize / meanShapeSize;
  // cv::Mat scaledMeanShape = meanShape * scale;

  cv::Rect_<float> initLandmarkMeanRect;
  boundingRect(initLandmark, initLandmarkMeanRect);

  float rotationStdDevRad = rotationStdDev_ * CV_PI / 180;
  float translationXStdDev = translationMultX_ * initLandmarkMeanRect.width;
  float translationYStdDev = translationMultY_ * initLandmarkMeanRect.height;

  cv::Mat tempInit = initLandmark.clone();

  float rotation_prob, offsetX_prob, offsetY_prob, scaling_prob;
  float angle2 = 0., offsetX = 0., offsetY = 0., scaling = 1.;
  if (random_rotation_)
  {
    caffe_rng_uniform<float>(1, 0.f, 1.f, &rotation_prob);
    if (rotation_prob < rotation_prob_)
    {
      //caffe_rng_gaussian<float>(1, 0.f, rotationStdDevRad, &angle2);
      caffe_rng_uniform<float>(1, -rotationStdDevRad, rotationStdDevRad, &angle2);
    }
  }
  else
  {
    angle2 = 0.;
  }

  if (random_translationX_)
  {
    caffe_rng_uniform<float>(1, 0.f, 1.f, &offsetX_prob);
    if (offsetX_prob < translationX_prob_)
    {
      //caffe_rng_gaussian<float>(1, 0.f, translationXStdDev, &offsetX);
      caffe_rng_uniform<float>(1, -translationXStdDev, translationXStdDev, &offsetX);
    }
  }
  else
  {
    offsetX = 0.;
  }

  if (random_translationY_)
  {
    caffe_rng_uniform<float>(1, 0.f, 1.f, &offsetY_prob);
    if (offsetY_prob < translationY_prob_)
    {
      //caffe_rng_gaussian<float>(1, 0.f, translationYStdDev, &offsetY);
      caffe_rng_uniform<float>(1, -translationYStdDev, translationYStdDev, &offsetY);
    }
  }
  else
  {
    offsetY = 0.;
  }
  if (random_scale_)
  {
    caffe_rng_uniform<float>(1, 0.f, 1.f, &scaling_prob);
    if (scaling_prob < scale_prob_)
    {
      //caffe_rng_gaussian<float>(1, 1.f, scaleStdDev_, &scaling);
      caffe_rng_uniform<float>(1, 1.f - scaleStdDev_ / 2., 1.f + scaleStdDev_, &scaling);
    }
  }
  else
  {
    scaling = 1.;
  }

  for (int i = 0; i < tempInit.rows; ++i)
  {
    tempInit.at<float>(i, 0) += offsetX;
    tempInit.at<float>(i, 1) += offsetY;
  }
  float tempInitX, tempInitY;
  meanXYOfShape(tempInit, tempInitX, tempInitY);
  for (int i = 0; i < tempInit.rows; ++i)
  {
    float x, y;
    x = tempInit.at<float>(i, 0);
    y = tempInit.at<float>(i, 1);
    tempInit.at<float>(i, 0) = (x - tempInitX) * scaling + tempInitX;
    tempInit.at<float>(i, 1) = (y - tempInitY) * scaling + tempInitY;
  }

  meanXYOfShape(tempInit, tempInitX, tempInitY);
  for (int i = 0; i < tempInit.rows; ++i)
  {
    tempInit.at<float>(i, 0) -= tempInitX;
    tempInit.at<float>(i, 1) -= tempInitY;
  }
  cv::Mat R(2, 2, CV_32FC1);
  R.at<float>(0, 0) = cosf(angle2);
  R.at<float>(0, 1) = sinf(angle2);
  R.at<float>(1, 0) = -sinf(angle2);
  R.at<float>(1, 1) = cosf(angle2);
  cv::Mat tempInit2;
  cv::gemm(tempInit, R.t(), 1.f, cv::Mat(), 0.f, tempInit2);
  for (int i = 0; i < tempInit2.rows; ++i)
  {
    tempInit2.at<float>(i, 0) += tempInitX;
    tempInit2.at<float>(i, 1) += tempInitY;
  }

  angle = angle2 / CV_PI * 180.;
  return tempInit2;
}

template <typename Dtype>
bool FaceAlignDataLayer<Dtype>::cropResizeRotate(
    const cv::Mat &meanShape,
    const cv::Mat &image,
    const cv::Mat &initLandmark,
    const cv::Mat &gtLandmark,
    cv::Mat &tempImg,
    cv::Mat &tempInit,
    cv::Mat &tempGroundTruth)
{
  cv::Rect_<float> meanShapeRect;
  boundingRect(meanShape, meanShapeRect);
  float meanShapeSize = std::max(meanShapeRect.height, meanShapeRect.width);
  float destShapeSize = new_image_size_ * (1 - 2. * frame_fraction_);

  float scale = destShapeSize / meanShapeSize;
  cv::Mat scaledMeanShape = meanShape * scale;
  float scaledMeanX, scaledMeanY;
  meanXYOfShape(scaledMeanShape, scaledMeanX, scaledMeanY);
  cv::Mat destShape = scaledMeanShape.clone();
  for (int i = 0; i < destShape.rows; ++i)
  {
    destShape.at<float>(i, 0) += -scaledMeanX + static_cast<float>(new_image_size_) / 2.;
    destShape.at<float>(i, 1) += -scaledMeanY + static_cast<float>(new_image_size_) / 2.;
  }

  cv::Mat A, t;
  bestFitMat(destShape, initLandmark, A, t);
  cv::Mat A2 = A.inv();
  A2.convertTo(A2, CV_32F);
  cv::Mat t2 = -t;
  cv::gemm(t2, A2, 1.f, cv::Mat(), 0.f, t2);
  // t2 = t2.dot(A2);
  cv::Mat M(2, 3, CV_32FC1);
  M.at<float>(0, 0) = A2.at<float>(0, 0);
  M.at<float>(0, 1) = A2.at<float>(1, 0);
  M.at<float>(1, 0) = A2.at<float>(0, 1);
  M.at<float>(1, 1) = A2.at<float>(1, 1);
  M.at<float>(0, 2) = t2.at<float>(0, 0);
  M.at<float>(1, 2) = t2.at<float>(0, 1);

  cv::warpAffine(image, tempImg, M, cv::Size(new_image_size_, new_image_size_),
                 cv::INTER_LINEAR | cv::WARP_INVERSE_MAP);
  cv::gemm(initLandmark, A, 1.f, cv::Mat(), 0.f, tempInit);
  for (int i = 0; i < tempInit.rows; ++i)
  {
    tempInit.at<float>(i, 0) += t.at<float>(0, 0);
    tempInit.at<float>(i, 1) += t.at<float>(0, 1);
  }
  // tempInit = initLandmark.dot(A) + t;
  cv::gemm(gtLandmark, A, 1.f, cv::Mat(), 0.f, tempGroundTruth);
  for (int i = 0; i < tempGroundTruth.rows; ++i)
  {
    tempGroundTruth.at<float>(i, 0) += t.at<float>(0, 0);
    tempGroundTruth.at<float>(i, 1) += t.at<float>(0, 1);
  }
  // tempGroundTruth += t;
  // tempGroundTruth = tempGroundTruth.dot(A) + t;

  cv::Rect_<float> valid_rect(0, 0, new_image_size_, new_image_size_);
  bool flag = true;
  int valid_count = 0;
  for (int i = 0; i < tempGroundTruth.rows; ++i)
  {
    float x = tempGroundTruth.at<float>(i, 0);
    float y = tempGroundTruth.at<float>(i, 1);
    if (!valid_rect.contains(cv::Point_<float>(x, y)))
    {
      flag = false;
      break;
    }
    else
    {
      ++valid_count;
    }
  }
  if (!flag)
  {
    return false;
  }
  return true;
}

template <typename Dtype>
cv::Mat FaceAlignDataLayer<Dtype>::bestFitRect(const cv::Mat &groundTruth, const cv::Mat &meanShape)
{
  cv::Point2f box_pt1(groundTruth.at<float>(0, 0), groundTruth.at<float>(0, 1));
  cv::Point2f box_pt2(groundTruth.at<float>(0, 0), groundTruth.at<float>(0, 1));

  cv::Point2f box2_pt1(meanShape.at<float>(0, 0), meanShape.at<float>(0, 1));
  cv::Point2f box2_pt2(meanShape.at<float>(0, 0), meanShape.at<float>(0, 1));
  for (int i = 0; i < groundTruth.rows; ++i)
  {
    float x = groundTruth.at<float>(i, 0);
    float y = groundTruth.at<float>(i, 1);
    if (x < box_pt1.x)
    {
      box_pt1.x = x;
    }
    if (y < box_pt1.y)
    {
      box_pt1.y = y;
    }
    if (x > box_pt2.x)
    {
      box_pt2.x = x;
    }
    if (y > box_pt2.y)
    {
      box_pt2.y = y;
    }
    x = meanShape.at<float>(i, 0);
    y = meanShape.at<float>(i, 1);
    if (x < box2_pt1.x)
    {
      box2_pt1.x = x;
    }
    if (y < box2_pt1.y)
    {
      box2_pt1.y = y;
    }
    if (x > box2_pt2.x)
    {
      box2_pt2.x = x;
    }
    if (y > box2_pt2.y)
    {
      box2_pt2.y = y;
    }
  }
  cv::Point2f boxCenter((box_pt1.x + box_pt2.x) / 2., (box_pt1.y + box_pt2.y) / 2.);
  float boxWidth = box_pt2.x - box_pt1.x;
  float boxHeight = box_pt2.y - box_pt1.y;
  float meanShapeWidth = box2_pt2.x - box2_pt1.x;
  float meanShapeHeight = box2_pt2.y - box2_pt1.y;

  float scaleWidth = boxWidth / meanShapeWidth;
  float scaleHeight = boxHeight / meanShapeHeight;
  float scale = (scaleWidth + scaleHeight) / 2;
  cv::Mat S0 = meanShape * scale;

  cv::Point2f box3_pt1(S0.at<float>(0, 0), S0.at<float>(0, 1));
  cv::Point2f box3_pt2(S0.at<float>(0, 0), S0.at<float>(0, 1));
  for (int i = 0; i < S0.rows; ++i)
  {
    float x = S0.at<float>(i, 0);
    float y = S0.at<float>(i, 1);
    if (x < box3_pt1.x)
    {
      box3_pt1.x = x;
    }
    if (y < box3_pt1.y)
    {
      box3_pt1.y = y;
    }
    if (x > box3_pt2.x)
    {
      box3_pt2.x = x;
    }
    if (y > box3_pt2.y)
    {
      box3_pt2.y = y;
    }
  }
  cv::Point2f S0Center((box3_pt1.x + box3_pt2.x) / 2., (box3_pt1.y + box3_pt2.y) / 2.);
  cv::Point2f diffCenter = boxCenter - S0Center;
  for (int i = 0; i < S0.rows; ++i)
  {
    S0.at<float>(i, 0) += diffCenter.x;
    S0.at<float>(i, 1) += diffCenter.y;
  }
  return S0;
}

template <typename Dtype>
cv::Mat FaceAlignDataLayer<Dtype>::mirrorShape(const cv::Mat &groundTruth, const cv::Mat &image)
{
  int width = image.cols;
  cv::Mat s = groundTruth.clone();
  float x, y;
  if (groundTruth.rows == 5)
  {
    for (int i = 0; i < 5; ++i)
    {
      s.at<float>(i, 0) = width - s.at<float>(i, 0);
    }
    // swap 1 and 2
    x = s.at<float>(0, 0);
    y = s.at<float>(0, 1);
    s.at<float>(0, 0) = s.at<float>(1, 0);
    s.at<float>(0, 1) = s.at<float>(1, 1);
    s.at<float>(1, 0) = x;
    s.at<float>(1, 1) = y;

    // swap 4 and 5
    x = s.at<float>(3, 0);
    y = s.at<float>(3, 1);
    s.at<float>(3, 0) = s.at<float>(4, 0);
    s.at<float>(3, 1) = s.at<float>(4, 1);
    s.at<float>(4, 0) = x;
    s.at<float>(4, 1) = y;
  }
  else
  {
    CHECK(false) << "do not support number of landmark: " << groundTruth.rows;
  }
  return s;
}

template <typename Dtype>
void FaceAlignDataLayer<Dtype>::boundingRect(const cv::Mat &shape, cv::Rect_<float> &rect)
{
  cv::Point2f box_pt1(shape.at<float>(0, 0), shape.at<float>(0, 1));
  cv::Point2f box_pt2(shape.at<float>(0, 0), shape.at<float>(0, 1));
  for (int i = 0; i < shape.rows; ++i)
  {
    float x = shape.at<float>(i, 0);
    float y = shape.at<float>(i, 1);
    if (x < box_pt1.x)
    {
      box_pt1.x = x;
    }
    if (y < box_pt1.y)
    {
      box_pt1.y = y;
    }
    if (x > box_pt2.x)
    {
      box_pt2.x = x;
    }
    if (y > box_pt2.y)
    {
      box_pt2.y = y;
    }
  }
  rect.x = box_pt1.x;
  rect.y = box_pt1.y;
  rect.width = box_pt2.x - box_pt1.x;
  rect.height = box_pt2.y - box_pt1.y;
}

template <typename Dtype>
void FaceAlignDataLayer<Dtype>::meanXYOfShape(const cv::Mat &shape, float &x, float &y)
{
  x = 0.f;
  y = 0.f;
  for (int i = 0; i < shape.rows; ++i)
  {
    x += shape.at<float>(i, 0);
    y += shape.at<float>(i, 1);
  }
  x /= static_cast<float>(shape.rows);
  y /= static_cast<float>(shape.rows);
}

template <typename Dtype>
void FaceAlignDataLayer<Dtype>::bestFitMat(
    const cv::Mat &destination,
    const cv::Mat &source,
    cv::Mat &A,
    cv::Mat &t)
{
  vector<float> dest(destination.rows * 2);
  vector<float> src(source.rows * 2);
  for (int i = 0; i < destination.rows; ++i)
  {
    dest[2 * i + 0] = destination.at<float>(i, 0);
    dest[2 * i + 1] = destination.at<float>(i, 1);

    src[2 * i + 0] = source.at<float>(i, 0);
    src[2 * i + 1] = source.at<float>(i, 1);
  }
  vector<float> transformed_src(source.rows * 2);
  vector<float> T(6);
  bestFit<float>(destination.rows, dest.data(), src.data(), &transformed_src[0], &T[0]);
  A.create(2, 2, CV_32FC1);
  t.create(1, 2, CV_32FC1);
  A.at<float>(0, 0) = T[2 * 0 + 0];
  A.at<float>(0, 1) = T[2 * 0 + 1];
  A.at<float>(1, 0) = T[2 * 1 + 0];
  A.at<float>(1, 1) = T[2 * 1 + 1];
  t.at<float>(0, 0) = T[2 * 2 + 0];
  t.at<float>(0, 1) = T[2 * 2 + 1];
}

template <typename Dtype>
void bestFit(const int num_landmark, const Dtype *dest, const Dtype *src, Dtype *transformed_src, Dtype *T)
{
  Dtype dstMean_x = 0, dstMean_y = 0, srcMean_x = 0, srcMean_y = 0;
  for (size_t i = 0; i < num_landmark; i++)
  {
    dstMean_x += dest[2 * i + 0];
    dstMean_y += dest[2 * i + 1];
    srcMean_x += src[2 * i + 0];
    srcMean_y += src[2 * i + 1];
  }
  dstMean_x /= static_cast<Dtype>(num_landmark);
  dstMean_y /= static_cast<Dtype>(num_landmark);
  srcMean_x /= static_cast<Dtype>(num_landmark);
  srcMean_y /= static_cast<Dtype>(num_landmark);

  vector<Dtype> srcVec(2 * num_landmark);
  vector<Dtype> dstVec(2 * num_landmark);
  for (size_t i = 0; i < num_landmark; i++)
  {
    srcVec[2 * i + 0] = src[2 * i + 0] - srcMean_x;
    srcVec[2 * i + 1] = src[2 * i + 1] - srcMean_y;
    dstVec[2 * i + 0] = dest[2 * i + 0] - dstMean_x;
    dstVec[2 * i + 1] = dest[2 * i + 1] - dstMean_y;
  }
  Dtype srcVecL2Norm = caffe_l2_norm(2 * num_landmark, srcVec.data());
  Dtype a = caffe_cpu_dot(2 * num_landmark, srcVec.data(), dstVec.data()) / (srcVecL2Norm * srcVecL2Norm);
  Dtype b = Dtype(0);
  for (size_t i = 0; i < num_landmark; i++)
  {
    b += srcVec[2 * i] * dstVec[2 * i + 1] - srcVec[2 * i + 1] * dstVec[2 * i];
  }
  b /= srcVecL2Norm * srcVecL2Norm;
  T[0] = a;
  T[1] = -b;
  T[2] = b;
  T[3] = a;
  T[4] = dstMean_x - (srcMean_x * T[0] + srcMean_y * T[2]);
  T[5] = dstMean_y - (srcMean_x * T[1] + srcMean_y * T[3]);
  if (transformed_src != NULL)
  {
    for (size_t i = 0; i < num_landmark; i++)
    {
      transformed_src[2 * i + 0] = dstMean_x;
      transformed_src[2 * i + 1] = dstMean_y;
    }
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
                          num_landmark, 2, 2, (Dtype)1.,
                          srcVec.data(), T, (Dtype)1., transformed_src);
  }
}

INSTANTIATE_CLASS(FaceAlignDataLayer);
REGISTER_LAYER_CLASS(FaceAlignData);

} // namespace caffe
