#include <opencv2/core/core.hpp>

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
  // Read a data point, and use it to initialize the top blob.
  DrivingData data;
  data.ParseFromString(*(this->reader_.full().peek()));
  const Datum &datum = data.car_image_datum();

  vector<int> top_shape(4,0);
  int shape[4] = {batch_size, datum.channels(),
                  data.car_cropped_height(), data.car_cropped_width()};
  memcpy(&top_shape[0], shape, sizeof(shape));

  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    vector<int> label_shape(4,0);
    int shape[4] = {
        batch_size, kNumRegressionMasks,
        data.car_label_height() * data.car_label_resolution(),
        data.car_label_width() * data.car_label_resolution()
    };
    memcpy(&label_shape[0], shape, sizeof(shape));
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(label_shape);
    }
  }
}

int Rand() {return rand();}
float rand_float() {return rand()*1.0f / RAND_MAX;}

bool ReadBoundingBoxLabelToDatum(
    const DrivingData& data, Datum* datum, const int h_off, const int w_off,
    const DriveDataParameter& param) {
  bool have_obj = false;
  const int grid_dim = data.car_label_resolution();
  const int width = data.car_label_width();
  const int height = data.car_label_height();
  const int full_label_width = width * grid_dim;
  const int full_label_height = height * grid_dim;
  const float half_shrink_factor = (1-param.shrink_prob_factor()) / 2;
  const float unrecog_factor = param.unrecognize_factor();
  const float scaling = static_cast<float>(full_label_width) \
    / data.car_cropped_width();

  // 1 pixel label, 4 bounding box coordinates, 3 normalization labels.
  const int num_total_labels = kNumRegressionMasks;
  vector<cv::Mat *> labels;
  for (int i = 0; i < num_total_labels; ++i) {
    labels.push_back(
        new cv::Mat(full_label_height,
                    full_label_width, CV_32F,
                    cv::Scalar(0.0)));
  }

  for (int i = 0; i < data.car_boxes_size(); ++i) {
    int xmin = data.car_boxes(i).xmin();
    int ymin = data.car_boxes(i).ymin();
    int xmax = data.car_boxes(i).xmax();
    int ymax = data.car_boxes(i).ymax();
    float ow = xmax - xmin;
    float oh = ymax - ymin;
    xmin = std::min(std::max(0, xmin - w_off), data.car_cropped_width());
    xmax = std::min(std::max(0, xmax - w_off), data.car_cropped_width());
    ymin = std::min(std::max(0, ymin - h_off), data.car_cropped_height());
    ymax = std::min(std::max(0, ymax - h_off), data.car_cropped_height());
    float w = xmax - xmin;
    float h = ymax - ymin;
    // drop boxes that unrecognize
    if (w < ow*unrecog_factor || h < oh*unrecog_factor)
        continue;
    if (w < 4 || h < 4) {
      // drop boxes that are too small
      continue;
    }
    have_obj = true;
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

  CHECK_EQ(datum->float_data_size(),
           num_total_labels * full_label_height * full_label_width);
  for (int i = 0; i < num_total_labels; ++i) {
    delete labels[i];
  }

  return have_obj;
}

// This function is called on prefetch thread
template<typename Dtype>
void DriveDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  DrivingData data;
  string *raw_data;
  data.ParseFromString(*(raw_data = this->reader_.full().pop("Waiting for data")));
  const Datum& datum = data.car_image_datum();

  vector<int> top_shape(4,0);
  int shape[4] = {batch_size, datum.channels(),
                  data.car_cropped_height(), data.car_cropped_width()};
  memcpy(&top_shape[0], shape, sizeof(shape));

  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* top_data = batch->data_.mutable_cpu_data();
  const Dtype* data_mean = this->data_transformer_->data_mean_.cpu_data();
  vector<Dtype*> top_labels;

  if (this->output_labels_) {
    Dtype *top_label = batch->label_.mutable_cpu_data();
    top_labels.push_back(top_label);
  }

  const int crop_num = this->layer_param().drive_data_param().crop_num();

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a datum
    if (item_id != 0 && item_id%crop_num == 0)
        data.ParseFromString(*(raw_data = this->reader_.full().pop("Waiting for data")));
    read_time += timer.MicroSeconds();
    timer.Start();

    const Datum& img_datum = data.car_image_datum();
    const string& img_datum_data = img_datum.data();
    bool can_pass = rand_float() > this->layer_param().drive_data_param().random_crop_ratio();
    try_again:
    int h_off = img_datum.height() == data.car_cropped_height() ?
        0 : Rand() % (img_datum.height() - data.car_cropped_height());
    int w_off = img_datum.width() == data.car_cropped_width() ?
        0 : Rand() % (img_datum.width() - data.car_cropped_width());

    vector<Datum> label_datums(kNumLabels);
    if (this->output_labels_) {
      // Call appropriate functions for genearting each label
      if (!ReadBoundingBoxLabelToDatum(data, &label_datums[0],
            h_off, w_off, this->layer_param().drive_data_param()))
          if (can_pass)
            goto try_again;
    }

    for (int c = 0; c < img_datum.channels(); ++c) {
      for (int h = 0; h < data.car_cropped_height(); ++h) {
        for (int w = 0; w < data.car_cropped_width(); ++w) {
          int top_index = ((item_id * img_datum.channels() + c) \
                           * data.car_cropped_height() + h)
              * data.car_cropped_width() + w;
          int data_index = (c * img_datum.height() + h + h_off) \
            * img_datum.width() + w + w_off;
          uint8_t datum_element_ui8 = \
            static_cast<uint8_t>(img_datum_data[data_index]);
          Dtype datum_element = static_cast<Dtype>(datum_element_ui8);

          top_data[top_index] = datum_element - data_mean[data_index];
        }
      }
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

    if (item_id%crop_num == 0)
        this->reader_.free().push(const_cast<string*>(raw_data));
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
