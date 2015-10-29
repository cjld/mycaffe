#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

const int kNumData = 1;
const int kNumLabels = 1;
const int kNumBBRegressionCoords = 4;
const int kNumRegressionMasks = 8;

template <typename Dtype>
DriveDataLayer<Dtype>::DriveDataLayer(const LayerParameter& param)
  : DataLayer<Dtype>(param) {
}

template <typename Dtype>
void DriveDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  srand(0);
  const int batch_size = this->layer_param_.data_param().batch_size();
  DriveDataParameter param = this->layer_param().drive_data_param();
  // Read a data point, and use it to initialize the top blob.

  vector<int> top_shape(4,0);
  int shape[4] = {batch_size, 3,
                  (int)param.cropped_height(), (int)param.cropped_width()};
  memcpy(&top_shape[0], shape, sizeof(shape));

  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data().Reshape(top_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    vector<int> label_shape(4,0);
    vector<int> type_shape(4,0);
    int shape[4] = {
        batch_size, kNumRegressionMasks,
        param.tiling_height() * param.label_resolution(),
        param.tiling_width() * param.label_resolution()
    };
    int shape_type[4] = {
        batch_size, 1,
        param.tiling_height() * param.catalog_resolution(),
        param.tiling_width() * param.catalog_resolution()
    };
    memcpy(&label_shape[0], shape, sizeof(shape));
    top[1]->Reshape(label_shape);

    memcpy(&type_shape[0], shape_type, sizeof(shape_type));
    CHECK_GE(top.size(), 3);
    top[2]->Reshape(type_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label(0).Reshape(label_shape);
      this->prefetch_[i].label(1).Reshape(type_shape);
    }

  }
}

int Rand() {return rand();}
float rand_float() {return rand()*1.0f / RAND_MAX;}

bool ReadBoundingBoxLabelToDatum(
    const DrivingData& data, Datum* datum, const int h_off, const int w_off, const float resize,
    const DriveDataParameter& param, float* label_type, bool can_pass, int bid) {
  bool have_obj = false;
  const int grid_dim = param.label_resolution();
  const int width = param.tiling_width();
  const int height = param.tiling_height();
  const int full_label_width = width * grid_dim;
  const int full_label_height = height * grid_dim;
  const float half_shrink_factor = (1-param.shrink_prob_factor()) / 2;
  const float unrecog_factor = param.unrecognize_factor();
  const float scaling = static_cast<float>(full_label_width) \
    / param.cropped_width();
  //const float resize = param.resize();

  const int type_label_width = width * param.catalog_resolution();
  const int type_label_height = height * param.catalog_resolution();
  const int type_stride = full_label_width / type_label_width;

  // fast check

  for (int i = 0; i < data.car_boxes_size(); ++i) {
    if (i != bid) continue;
    float xmin = data.car_boxes(i).xmin()*resize;
    float ymin = data.car_boxes(i).ymin()*resize;
    float xmax = data.car_boxes(i).xmax()*resize;
    float ymax = data.car_boxes(i).ymax()*resize;
    assert(ttype+1 < param.catalog_number());
    float ow = xmax - xmin;
    float oh = ymax - ymin;
    xmin = std::min<float>(std::max<float>(0, xmin - w_off), param.cropped_width());
    xmax = std::min<float>(std::max<float>(0, xmax - w_off), param.cropped_width());
    ymin = std::min<float>(std::max<float>(0, ymin - h_off), param.cropped_height());
    ymax = std::min<float>(std::max<float>(0, ymax - h_off), param.cropped_height());
    float w = xmax - xmin;
    float h = ymax - ymin;
    // drop boxes that unrecognize
    if (w*h < ow*oh*unrecog_factor)
        continue;

    if (w < 4 || h < 4) {
      // drop boxes that are too small
      continue;
    }
    if (std::max(w,h) < param.train_min() || std::max(w,h) > param.train_max())
        continue;
    have_obj = true;
  }
  if (can_pass && !have_obj) return false;


  // 1 pixel label, 4 bounding box coordinates, 3 normalization labels.
  const int num_total_labels = kNumRegressionMasks;
  cv::Mat box_mask(full_label_height,
                   full_label_width, CV_32F,
                   cv::Scalar(0.0));
  //cv::Mat circle_mask, poly_mask;
  vector<cv::Mat *> labels;
  for (int i = 0; i < num_total_labels; ++i) {
    labels.push_back(
        new cv::Mat(full_label_height,
                    full_label_width, CV_32F,
                    cv::Scalar(0.0)));
  }
  for (int i = 0; i < data.car_boxes_size(); ++i) {
    float xmin = data.car_boxes(i).xmin()*resize;
    float ymin = data.car_boxes(i).ymin()*resize;
    float xmax = data.car_boxes(i).xmax()*resize;
    float ymax = data.car_boxes(i).ymax()*resize;
    int ttype = data.car_boxes(i).type();
    assert(ttype+1 < param.catalog_number());
    float ow = xmax - xmin;
    float oh = ymax - ymin;
    xmin = std::min<float>(std::max<float>(0, xmin - w_off), param.cropped_width());
    xmax = std::min<float>(std::max<float>(0, xmax - w_off), param.cropped_width());
    ymin = std::min<float>(std::max<float>(0, ymin - h_off), param.cropped_height());
    ymax = std::min<float>(std::max<float>(0, ymax - h_off), param.cropped_height());
    float w = xmax - xmin;
    float h = ymax - ymin;
    // drop boxes that unrecognize
    if (w*h < ow*oh*unrecog_factor)
        continue;
    if (w < 4 || h < 4) {
      // drop boxes that are too small
      continue;
    }

    if (std::max(w,h) < param.reco_min() || std::max(w,h) > param.reco_max())
        continue;
    // shrink bboxes
    int gxmin = cvFloor((xmin + w * half_shrink_factor) * scaling);
    int gxmax = cvCeil((xmax - w * half_shrink_factor) * scaling);
    int gymin = cvFloor((ymin + h * half_shrink_factor) * scaling);
    int gymax = cvCeil((ymax - h * half_shrink_factor) * scaling);

    CHECK_LE(gxmin, gxmax);
    CHECK_LE(gymin, gymax);
    if (gxmin >= full_label_width) {
      gxmin = full_label_width - 1;
    }
    if (gymin >= full_label_height) {
      gymin = full_label_height - 1;
    }
    CHECK_LE(0, gxmin);
    CHECK_LE(0, gymin);
    CHECK_LE(gxmax, full_label_width);
    CHECK_LE(gymax, full_label_height);
    if (gxmin == gxmax) {
      if (gxmax < full_label_width - 1) {
        gxmax++;
      } else if (gxmin > 0) {
        gxmin--;
      }
    }
    if (gymin == gymax) {
      if (gymax < full_label_height - 1) {
        gymax++;
      } else if (gymin > 0) {
        gymin--;
      }
    }
    CHECK_LT(gxmin, gxmax);
    CHECK_LT(gymin, gymax);
    if (gxmax == full_label_width) {
      gxmax--;
    }
    if (gymax == full_label_height) {
      gymax--;
    }
    cv::Rect r(gxmin, gymin, gxmax - gxmin + 1, gymax - gymin + 1);

    float flabels[num_total_labels] =
        {1.0f, (float)xmin, (float)ymin, (float)xmax, (float)ymax, 1.0f / w, 1.0f / h, 1.0f};
    for (int j = 0; j < num_total_labels; ++j) {
      cv::Mat roi(*labels[j], r);
      roi = cv::Scalar(flabels[j]);
    }
    cv::rectangle(box_mask, r, cv::Scalar(ttype+1), -1);
  }

  int total_num_pixels = 0;
  for (int y = 0; y < full_label_height; ++y) {
    for (int x = 0; x < full_label_width; ++x) {
      if (labels[num_total_labels - 1]->at<float>(y, x) == 1.0) {
        total_num_pixels++;
      }
    }
  }
  if (total_num_pixels != 0) {
    float reweight_value = 1.0 / total_num_pixels;
    for (int y = 0; y < full_label_height; ++y) {
      for (int x = 0; x < full_label_width; ++x) {
        if (labels[num_total_labels - 1]->at<float>(y, x) == 1.0) {
          labels[num_total_labels - 1]->at<float>(y, x) = reweight_value;
        }
      }
    }
  }

  datum->set_channels(num_total_labels);
  datum->set_height(full_label_height);
  datum->set_width(full_label_width);
  datum->set_label(0);  // dummy label
  datum->clear_data();
  datum->clear_float_data();

  for (int m = 0; m < num_total_labels; ++m) {
    for (int y = 0; y < full_label_height; ++y) {
      for (int x = 0; x < full_label_width; ++x) {
        float adjustment = 0;
        float val = labels[m]->at<float>(y, x);
        if (m == 0 || m > 4) {
          // do nothing
        } else if (labels[0]->at<float>(y, x) == 0.0) {
          // do nothing
        } else if (m % 2 == 1) {
          // x coordinate
          adjustment = x / scaling;
        } else {
          // y coordinate
          adjustment = y / scaling;
        }
        datum->add_float_data(val - adjustment);
      }
    }
  }

  // handle catalog
  float *ptr = label_type;
  for (int y = 0; y < type_label_height; ++y) {
    for (int x = 0; x < type_label_width; ++x) {
      *ptr = box_mask.at<float>(y*type_stride,x*type_stride);
      ptr++;
    }
  }

  CHECK_EQ(datum->float_data_size(),
           num_total_labels * full_label_height * full_label_width);

  for (int i = 0; i < num_total_labels; ++i) {
    delete labels[i];
  }
  return have_obj;
}

float get_box_size(const caffe::CarBoundingBox &box) {
    float xmin = box.xmin();
    float ymin = box.ymin();
    float xmax = box.xmax();
    float ymax = box.ymax();
    return std::max(xmax-xmin, ymax-ymin);
}

// This function is called on prefetch thread
template<typename Dtype>
void DriveDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data().count());
  DriveDataParameter param = this->layer_param_.drive_data_param();

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  DrivingData data;
  string *raw_data = NULL;

  vector<int> top_shape(4,0);
  int shape[4] = {batch_size, 3,
                  (int)param.cropped_height(), (int)param.cropped_width()};
  memcpy(&top_shape[0], shape, sizeof(shape));

  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data().Reshape(top_shape);

  Dtype* top_data = batch->data().mutable_cpu_data();
  const Dtype* data_mean_c = this->data_transformer_->data_mean_.cpu_data();
  vector<Dtype> v_mean(data_mean_c, data_mean_c+this->data_transformer_->data_mean_.count());
  Dtype *data_mean = &v_mean[0];

  vector<Dtype*> top_labels;
  Dtype *label_type = NULL;

  if (this->output_labels_) {
    Dtype *top_label = batch->label().mutable_cpu_data();
    top_labels.push_back(top_label);
    label_type = batch->label(1).mutable_cpu_data();
  }

  const int crop_num = this->layer_param().drive_data_param().crop_num();
  int type_label_strip = param.tiling_height()*param.tiling_width()
          *param.catalog_resolution()*param.catalog_resolution();
  bool need_new_data = true;
  int bid = 0;

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a datum
    if (need_new_data) {
        data.ParseFromString(*(raw_data = this->reader_.full().pop("Waiting for data")));
        bid = 0;
    } else {
        bid ++;
    }
    if (item_id+1 == batch_size) need_new_data = true;

    read_time += timer.MicroSeconds();
    timer.Start();
    Dtype *t_lable_type = NULL;
    if (label_type != NULL) {
        t_lable_type = label_type + type_label_strip*item_id;
        caffe_set(type_label_strip, (Dtype)0, t_lable_type);
    }
    vector<Datum> label_datums(kNumLabels);
    const Datum& img_datum = data.car_image_datum();
    const string& img_datum_data = img_datum.data();
    bool can_pass = rand_float() > this->layer_param().drive_data_param().random_crop_ratio();
    int cheight = param.cropped_height();
    int cwidth = param.cropped_width();
    int channal = img_datum.channels();
    int hid = bid/crop_num;
    float bsize = get_box_size(data.car_boxes(hid));
    float rmax = std::min(param.train_max() / bsize, param.resize_max());
    float rmin = std::max(param.train_min() / bsize, param.resize_min());

try_again:
    float resize = this->layer_param().drive_data_param().resize();
    int h_off, w_off;
    if (resize < 0) {
        if (rmax <= rmin || !can_pass) {
            can_pass = false;
            resize = rand_float() * (param.resize_max()-param.resize_min()) + param.resize_min();
            int rheight = (int)(img_datum.height() * resize);
            int rwidth = (int)(img_datum.height() * resize);
            h_off = rheight <= cheight ? 0 : Rand() % (rheight - cheight);
            w_off = rwidth <= cwidth ? 0 : Rand() % (rwidth - cwidth);
        } else {
            //can_pass = false;
            int h_max, h_min, w_max, w_min;
            resize = rand_float() * (rmax-rmin) + rmin;
            //LOG(INFO) << "resize " << rmax << " " << rmin << " " << resize << ' ' << item_id;
            h_min = data.car_boxes(hid).ymin()*resize - param.cropped_height();
            h_max = data.car_boxes(hid).ymax()*resize;
            w_min = data.car_boxes(hid).xmin()*resize - param.cropped_width();
            w_max = data.car_boxes(hid).xmax()*resize;
            w_off = (Rand() % (w_max-w_min) + w_min);
            h_off = (Rand() % (h_max-h_min) + h_min);
            //w_off = data.car_boxes(hid).xmax()*resize - param.cropped_width();;
            //h_off = data.car_boxes(hid).ymax()*resize - param.cropped_height();
        }
    } else {
        int rheight = (int)(img_datum.height() * resize);
        int rwidth = (int)(img_datum.height() * resize);
        h_off = rheight <= cheight ? 0 : Rand() % (rheight - cheight);
        w_off = rwidth <= cwidth ? 0 : Rand() % (rwidth - cwidth);
    }
    //LOG(INFO) << "?" << w_off << ' ' << h_off << ' ' << resize << ' ' << rmax << ' ' << rmin;


    if (this->output_labels_) {
      // Call appropriate functions for genearting each label
      if (!ReadBoundingBoxLabelToDatum(data, &label_datums[0],
            h_off, w_off, resize, param,(float*)t_lable_type, can_pass, hid))
          if (can_pass)
            goto try_again;
    }
    if (bid+1 >= crop_num*data.car_boxes_size())
        need_new_data = true;

    cv::Mat_<Dtype> mean_img(cheight, cwidth);
    Dtype* itop_data = top_data+item_id*channal*cheight*cwidth;
    float mat[] = { 1.f*resize,0.f,(float)-w_off, 0.f,1.f*resize,(float)-h_off };
    cv::Mat_<float> M(2,3, mat);

    for (int c=0; c<img_datum.channels(); c++) {
        cv::Mat_<Dtype> crop_img(cheight, cwidth, itop_data+c*cheight*cwidth);
        cv::Mat_<Dtype> pmean_img(img_datum.height(), img_datum.width(),
                          data_mean + c*img_datum.height()*img_datum.width());
        cv::Mat p_img(img_datum.height(), img_datum.width(), CV_8U,
               ((uint8_t*)&(img_datum_data[0])) + c*img_datum.height()*img_datum.width());
        p_img.convertTo(p_img, crop_img.type());
        cv::warpAffine(pmean_img, mean_img, M, mean_img.size(), cv::INTER_CUBIC);
        cv::warpAffine(p_img, crop_img, M, crop_img.size(), cv::INTER_CUBIC);

        crop_img -= mean_img;
        crop_img *= this->layer_param().drive_data_param().scale();
    }

    // Copy label.
    if (this->output_labels_) {
      for (int i = 0; i < kNumLabels; ++i) {
        for (int c = 0; c < label_datums[i].channels(); ++c) {
          for (int h = 0; h < label_datums[i].height(); ++h) {
            for (int w = 0; w < label_datums[i].width(); ++w) {
              const int top_index = ((item_id * label_datums[i].channels() + c)
                  * label_datums[i].height() + h) * label_datums[i].width() + w;
              const int data_index = (c * label_datums[i].height() + h) * \
                label_datums[i].width() + w;
              float label_datum_elem = label_datums[i].float_data(data_index);
              top_labels[i][top_index] = static_cast<Dtype>(label_datum_elem);
            }
          }
        }
      }
    }
    trans_time += timer.MicroSeconds();

    if (need_new_data) {
        this->reader_.free().push(const_cast<string*>(raw_data));
    }
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(DriveDataLayer);
REGISTER_LAYER_CLASS(DriveData);

}  // namespace caffe
