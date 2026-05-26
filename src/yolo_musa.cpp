/******************************************************************************\
|* YOLO11 Inference & mAP Evaluation (Optimized + FIXED NMS area order)      *|
|* Based on MThreads mtnn API (C++11)                                        *|
\******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <cmath>

#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>          // 用于 blobFromImage
#include <omp.h>
#include "mtnn_api.h"




// ---------- structures ----------
struct BBox {
    float x1, y1, x2, y2;   // pixel coordinates (xyxy)
    float conf;
    int   class_id;
};

struct ImageResult {
    std::string image_path;
    std::vector<BBox> detections;
    double inference_time_ms = 0.0;
    int img_width = 0;
    int img_height = 0;
};

struct GTBox {
    float x1, y1, x2, y2;   // pixel coordinates
    int   class_id;
};

// ---------- global settings ----------
static int   g_input_width      = 640;
static int   g_input_height     = 640;
static float g_conf_threshold   = 0.25f;
static float g_iou_threshold    = 0.5f;
static bool  g_compute_map      = false;
static std::string g_model_path;
static std::string g_input_path;
static std::string g_label_path;
static std::string g_output_path;
static int   g_num_classes     = 80;

// ---------- utility ----------
static double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static unsigned char* load_model(const char *filename, size_t *model_size) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) { fprintf(stderr, "Cannot open model %s\n", filename); return NULL; }
    fseek(fp, 0, SEEK_END);
    *model_size = ftell(fp);
    fclose(fp);
    unsigned char *data = (unsigned char*)malloc(*model_size);
    FILE *fp2 = fopen(filename, "rb");
    if (!fp2) { free(data); return NULL; }
    fread(data, 1, *model_size, fp2);
    fclose(fp2);
    return data;
}

static std::vector<std::string> list_images(const std::string &path) {
    std::vector<std::string> images;
    std::string clean_path = path;
    if (!clean_path.empty() && (clean_path.back() == '/' || clean_path.back() == '\\'))
        clean_path.pop_back();
    const char* exts[] = { "*.jpg", "*.jpeg", "*.png", "*.bmp" };
    for (int i = 0; i < 4; ++i) {
        std::string pattern = clean_path + "/" + exts[i];
        std::vector<cv::String> files;
        cv::glob(pattern, files, false);
        for (size_t j = 0; j < files.size(); ++j) images.push_back(files[j]);
    }
    printf("Found %zu images in %s\n", images.size(), clean_path.c_str());
    return images;
}

void preprocess(const cv::Mat& img, float* output)
{
    static cv::Mat blob;

    cv::dnn::blobFromImage(
        img,
        blob,
        1.0 / 255.0,
        cv::Size(g_input_width, g_input_height),
        cv::Scalar(),
        true,   // BGR->RGB
        false,  // no crop
        CV_32F
    );

    memcpy(output,
           blob.ptr<float>(),
           3 * g_input_width * g_input_height * sizeof(float));
}




// ---------- NMS（修正：面积计算必须在排序之后）----------
static std::vector<BBox> nms_global(std::vector<BBox> &detections, float iou_thres) {
    if (detections.empty()) return {};

    // 1. 按置信度降序原地稳定排序
    std::stable_sort(detections.begin(), detections.end(),
                     [](const BBox &a, const BBox &b) { return a.conf > b.conf; });

    // 2. 排序后再计算每个框的面积（与原算法一致）
    std::vector<float> areas(detections.size());
    for (size_t i = 0; i < detections.size(); ++i) {
        const auto &b = detections[i];
        areas[i] = (b.x2 - b.x1) * (b.y2 - b.y1);
    }

    // 3. 贪心 NMS，使用布尔向量标记抑制
    std::vector<bool> suppressed(detections.size(), false);
    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) continue;
        const auto &a = detections[i];
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j]) continue;
            const auto &b = detections[j];
            float ix1 = std::max(a.x1, b.x1);
            float iy1 = std::max(a.y1, b.y1);
            float ix2 = std::min(a.x2, b.x2);
            float iy2 = std::min(a.y2, b.y2);
            float iw = std::max(0.0f, ix2 - ix1);
            float ih = std::max(0.0f, iy2 - iy1);
            float inter = iw * ih;
            float iou = inter / (areas[i] + areas[j] - inter + 1e-6f);
            if (iou > iou_thres) {
                suppressed[j] = true;
            }
        }
    }

    // 4. 收集未被抑制的框
    std::vector<BBox> result;
    for (size_t i = 0; i < detections.size(); ++i) {
        if (!suppressed[i]) result.push_back(detections[i]);
    }
    return result;
}




static std::vector<BBox> process_output_fast(
    const float* output,
    int img_w,
    int img_h,
    float conf_thres,
    float iou_thres)
{
    constexpr int num_anchors = 8400;

    const int num_classes = g_num_classes;

    const float* cx_ptr = output;
    const float* cy_ptr = output + num_anchors;
    const float* w_ptr  = output + 2 * num_anchors;
    const float* h_ptr  = output + 3 * num_anchors;
    const float* cls_ptr = output + 4 * num_anchors;

    // 每个类别单独存储
    std::vector<std::vector<BBox>> class_boxes(num_classes);

    for (int c = 0; c < num_classes; ++c)
    {
        class_boxes[c].reserve(128);
    }

    // ====== 解码 ======
    #pragma omp parallel
    {
        std::vector<std::vector<BBox>> local_boxes(num_classes);

        for (int c = 0; c < num_classes; ++c)
        {
            local_boxes[c].reserve(64);
        }

        #pragma omp for nowait
        for (int i = 0; i < num_anchors; ++i)
        {
            float cx = cx_ptr[i];
            float cy = cy_ptr[i];
            float w  = w_ptr[i];
            float h  = h_ptr[i];

            int best_cls = -1;
            float best_score = conf_thres;

            // 找最大类别
            for (int c = 0; c < num_classes; ++c)
            {
                float score = cls_ptr[c * num_anchors + i];

                if (score > best_score)
                {
                    best_score = score;
                    best_cls = c;
                }
            }

            if (best_cls < 0)
                continue;

            BBox box;

            box.x1 = std::max(0.f,
                     (cx - w * 0.5f) * img_w / 640.f);

            box.y1 = std::max(0.f,
                     (cy - h * 0.5f) * img_h / 640.f);

            box.x2 = std::min((float)(img_w - 1),
                     (cx + w * 0.5f) * img_w / 640.f);

            box.y2 = std::min((float)(img_h - 1),
                     (cy + h * 0.5f) * img_h / 640.f);

            box.conf = best_score;
            box.class_id = best_cls;

            local_boxes[best_cls].push_back(box);
        }

        // merge（无锁优化）
        #pragma omp critical
        {
            for (int c = 0; c < num_classes; ++c)
            {
                auto& dst = class_boxes[c];
                auto& src = local_boxes[c];

                dst.insert(dst.end(),
                           src.begin(),
                           src.end());
            }
        }
    }

    // ====== Class-wise NMS ======
    std::vector<BBox> final_boxes;
    final_boxes.reserve(256);

    for (int c = 0; c < num_classes; ++c)
    {
        auto& boxes = class_boxes[c];

        if (boxes.empty())
            continue;

        // TopK（非常关键）
        constexpr int TOPK = 300;

        if ((int)boxes.size() > TOPK)
        {
            std::partial_sort(
                boxes.begin(),
                boxes.begin() + TOPK,
                boxes.end(),
                [](const BBox& a, const BBox& b)
                {
                    return a.conf > b.conf;
                });

            boxes.resize(TOPK);
        }
        else
        {
            std::sort(
                boxes.begin(),
                boxes.end(),
                [](const BBox& a, const BBox& b)
                {
                    return a.conf > b.conf;
                });
        }

        std::vector<int> keep;
        keep.reserve(boxes.size());

        std::vector<float> areas(boxes.size());

        for (size_t i = 0; i < boxes.size(); ++i)
        {
            areas[i] =
                (boxes[i].x2 - boxes[i].x1) *
                (boxes[i].y2 - boxes[i].y1);
        }

        for (size_t i = 0; i < boxes.size(); ++i)
        {
            bool suppressed = false;

            for (int kept_idx : keep)
            {
                const auto& a = boxes[i];
                const auto& b = boxes[kept_idx];

                float xx1 = std::max(a.x1, b.x1);
                float yy1 = std::max(a.y1, b.y1);
                float xx2 = std::min(a.x2, b.x2);
                float yy2 = std::min(a.y2, b.y2);

                float w = std::max(0.f, xx2 - xx1);
                float h = std::max(0.f, yy2 - yy1);

                float inter = w * h;

                float iou =
                    inter /
                    (areas[i] + areas[kept_idx] - inter + 1e-6f);

                if (iou > iou_thres)
                {
                    suppressed = true;
                    break;
                }
            }

            if (!suppressed)
            {
                keep.push_back(i);
            }
        }

        for (int idx : keep)
        {
            final_boxes.push_back(boxes[idx]);
        }
    }

    return final_boxes;
}








static std::vector<BBox> process_output(const float *output, int img_w, int img_h,
                                        float conf_thres, float iou_thres) {
    std::vector<BBox> detections;
    const int num_anchors = 8400;
    const int num_classes = g_num_classes;

    const float* cx_ptr = output;
    const float* cy_ptr = output + num_anchors;
    const float* w_ptr  = output + 2 * num_anchors;
    const float* h_ptr  = output + 3 * num_anchors;
    const float* cls_base = output + 4 * num_anchors;

    // 并行化锚框循环（每个线程独立处理部分锚框，再合并结果）
    #pragma omp parallel
    {
        std::vector<BBox> local_detections;  // 线程局部容器，避免锁竞争

        #pragma omp for nowait
        for (int i = 0; i < num_anchors; ++i) {
            float cx = cx_ptr[i];
            float cy = cy_ptr[i];
            float w  = w_ptr[i];
            float h  = h_ptr[i];

            float max_score = -1e9f;
            int best_cls = -1;
            for (int c = 0; c < num_classes; ++c) {
                float s = cls_base[c * num_anchors + i];
                if (s > max_score) { max_score = s; best_cls = c; }
            }
            if (max_score < conf_thres) continue;

            float x1 = (cx - w / 2) / 640.0f * img_w;
            float y1 = (cy - h / 2) / 640.0f * img_h;
            float x2 = (cx + w / 2) / 640.0f * img_w;
            float y2 = (cy + h / 2) / 640.0f * img_h;

            x1 = std::max(0.0f, std::min(x1, (float)img_w - 1));
            y1 = std::max(0.0f, std::min(y1, (float)img_h - 1));
            x2 = std::max(0.0f, std::min(x2, (float)img_w - 1));
            y2 = std::max(0.0f, std::min(y2, (float)img_h - 1));

            BBox b;
            b.x1 = x1; b.y1 = y1; b.x2 = x2; b.y2 = y2;
            b.conf = max_score;
            b.class_id = best_cls;
            local_detections.push_back(b);
        }

        // 合并各线程结果（临界区）
        #pragma omp critical
        {
            detections.insert(detections.end(),
                              local_detections.begin(), local_detections.end());
        }
    }

    return nms_global(detections, iou_thres);
}

// ---------- 单张图片推理（带分段计时） ----------
static ImageResult infer_image(const std::string &img_path,
                               mtnn_mgr &network,
                               const std::vector<mtnn_tensor_mem> &input_mem,
                               const std::vector<mtnn_tensor_attr> &out_attrs,
                               const std::vector<mtnn_tensor_mem> &out_mems,
                               bool debug_print = false) {
    ImageResult res;
    res.image_path = img_path;



    // 1. 图像读取
    double t_read0 = get_time_ms();
    cv::Mat img = cv::imread(img_path);
    double t_read1 = get_time_ms();
    if (img.empty()) {
        fprintf(stderr, "Failed to read image: %s\n", img_path.c_str());
        return res;
    }
    res.img_width = img.cols;
    res.img_height = img.rows;

     double t_total0 = get_time_ms();

    // 2. 前处理
	// 2. 前处理
	double t_pre0 = get_time_ms();
	
	float* input_ptr =
	    (float*)input_mem[0].logical_addr;
	
	preprocess(img, input_ptr);
	
	double t_pre1 = get_time_ms();

    // 3. 量化 + 输入拷贝（可能非常耗时）
    double t_quant0 = get_time_ms();

	// 改为直接拷贝浮点数据（假定设备内存足够容纳 float）
	//memcpy(input_mem[0].logical_addr, input_ptr, 128 * sizeof(float));




    
    double t_quant1 = get_time_ms();

    // 4. 设置输入并推理
    double t_infer_setup0 = get_time_ms();
    mtnn_input input_desc;
    memset(&input_desc, 0, sizeof(input_desc));
    input_desc.index = 0;
    input_desc.buf = input_mem[0].logical_addr;
    input_desc.size = input_mem[0].size;
    input_desc.pass_through = 1;
    input_desc.type = MTNN_TENSOR_FLOAT32;
    input_desc.fmt = MTNN_TENSOR_NCHW;

    int status = mtnn_inputs_set(network, 1, &input_desc);
    if (status != 0) {
        fprintf(stderr, "mtnn_inputs_set failed for %s\n", img_path.c_str());
        return res;
    }
    double t_infer_setup1 = get_time_ms();

    double t_infer0 = get_time_ms();
    status = mtnn_inference(network, NULL);
    double t_infer1 = get_time_ms();
    if (status != 0) {
        fprintf(stderr, "mtnn_inference failed for %s\n", img_path.c_str());
        return res;
    }

    // 5. 后处理（含 NMS）
    double t_post0 = get_time_ms();
    const float* output_ptr = (const float*)out_mems[0].logical_addr;

    // 可选的 debug 打印（仅第一张）
    if (debug_print) {
        printf("========== Debug: first image output ==========\n");
        printf("Output n_elems = %u\n", out_attrs[0].n_elems);
        printf("cx (first 10): ");
        for (int i = 0; i < 10; ++i) printf("%.4f ", output_ptr[i]);
        printf("\ncy (first 10): ");
        for (int i = 0; i < 10; ++i) printf("%.4f ", output_ptr[8400 + i]);
        printf("\nw  (first 10): ");
        for (int i = 0; i < 10; ++i) printf("%.4f ", output_ptr[2*8400 + i]);
        printf("\nh  (first 10): ");
        for (int i = 0; i < 10; ++i) printf("%.4f ", output_ptr[3*8400 + i]);
        printf("\ncls0 (first 10): ");
        for (int i = 0; i < 10; ++i) printf("%.4f ", output_ptr[4*8400 + i]);
        printf("\n================================================\n");
    }

    res.detections = process_output_fast(output_ptr, res.img_width, res.img_height,
                                    g_conf_threshold, g_iou_threshold);
    double t_post1 = get_time_ms();

    double t_total1 = get_time_ms();
    res.inference_time_ms = t_total1 - t_total0;

    // 打印分段耗时（每张图都打印，便于观察）
    printf("⏱️ [%s] timings (ms):\n", img_path.c_str());
    printf("   imread:      %.2f\n", t_read1 - t_read0);
    printf("   preprocess:   %.2f\n", t_pre1 - t_pre0);
    printf("   quant+copy:  %.2f  << check if large\n", t_quant1 - t_quant0);
    printf("   infer setup:  %.2f\n", t_infer_setup1 - t_infer_setup0);
    printf("   inference:    %.2f  (pure model run)\n", t_infer1 - t_infer0);
    printf("   postprocess: %.2f\n", t_post1 - t_post0);
    printf("   TOTAL:        %.2f\n", res.inference_time_ms);

    return res;
}

// ---------- 以下函数无变化 ----------
static std::vector<GTBox> load_labels(const std::string &label_file, int img_w, int img_h) {
    std::vector<GTBox> gts;
    std::ifstream f(label_file);
    if (!f.is_open()) return gts;
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream iss(line);
        float cls, xc, yc, w, h;
        if (!(iss >> cls >> xc >> yc >> w >> h)) continue;
        if (cls < 0 || cls >= g_num_classes) continue;
        float x1 = (xc - w/2) * img_w;
        float y1 = (yc - h/2) * img_h;
        float x2 = (xc + w/2) * img_w;
        float y2 = (yc + h/2) * img_h;
        GTBox gt;
        gt.x1 = x1; gt.y1 = y1; gt.x2 = x2; gt.y2 = y2;
        gt.class_id = (int)cls;
        gts.push_back(gt);
    }
    return gts;
}

static float compute_ap(const std::vector<std::pair<float, bool>> &pr, int total_gts) {
    if (pr.empty() || total_gts == 0) return 0.0f;

    std::vector<std::pair<float, bool>> sorted = pr;
    std::sort(sorted.begin(), sorted.end(),
              [](const std::pair<float, bool> &a, const std::pair<float, bool> &b) {
                  return a.first > b.first;
              });

    std::vector<float> recall, precision;
    int tp = 0, fp = 0;
    for (const auto &p : sorted) {
        if (p.second) tp++;
        else         fp++;
        recall.push_back((float)tp / total_gts);
        precision.push_back((float)tp / (tp + fp));
    }

    std::vector<float> mrec = {0.0f};
    mrec.insert(mrec.end(), recall.begin(), recall.end());
    mrec.push_back(1.0f);

    std::vector<float> mpre = {0.0f};
    mpre.insert(mpre.end(), precision.begin(), precision.end());
    mpre.push_back(0.0f);

    for (int i = (int)mpre.size() - 2; i >= 0; --i) {
        mpre[i] = std::max(mpre[i], mpre[i + 1]);
    }

    float ap = 0.0f;
    for (size_t i = 0; i < mrec.size() - 1; ++i) {
        if (mrec[i + 1] != mrec[i]) {
            ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1];
        }
    }
    return ap;
}

static float evaluate_map(const std::vector<ImageResult> &results,
                          const std::vector<std::vector<GTBox>> &all_gts,
                          float iou_thr) {
    std::vector<std::vector<std::pair<float, bool>>> class_pred(g_num_classes);
    std::vector<int> gt_counts(g_num_classes, 0);

    for (size_t i = 0; i < results.size(); ++i) {
        const auto &dets = results[i].detections;
        auto gts = all_gts[i];
        std::vector<bool> gt_matched(gts.size(), false);

        std::vector<BBox> sorted_dets = dets;
        std::sort(sorted_dets.begin(), sorted_dets.end(),
                  [](const BBox &a, const BBox &b) { return a.conf > b.conf; });

        for (const auto &d : sorted_dets) {
            int cls = d.class_id;
            if (cls < 0 || cls >= g_num_classes) continue;
            float best_iou = 0.0f;
            int best_j = -1;
            for (size_t j = 0; j < gts.size(); ++j) {
                if (gts[j].class_id != cls) continue;
                if (gt_matched[j]) continue;
                float inter_x1 = std::max(d.x1, gts[j].x1);
                float inter_y1 = std::max(d.y1, gts[j].y1);
                float inter_x2 = std::min(d.x2, gts[j].x2);
                float inter_y2 = std::min(d.y2, gts[j].y2);
                float iw = std::max(0.0f, inter_x2 - inter_x1);
                float ih = std::max(0.0f, inter_y2 - inter_y1);
                float inter = iw * ih;
                float area_d = (d.x2 - d.x1) * (d.y2 - d.y1);
                float area_g = (gts[j].x2 - gts[j].x1) * (gts[j].y2 - gts[j].y1);
                float iou = inter / (area_d + area_g - inter + 1e-6f);
                if (iou > best_iou) { best_iou = iou; best_j = j; }
            }
            if (best_iou >= iou_thr && best_j >= 0) {
                gt_matched[best_j] = true;
                class_pred[cls].push_back({d.conf, true});
            } else {
                class_pred[cls].push_back({d.conf, false});
            }
        }

        for (size_t j = 0; j < gts.size(); ++j)
            gt_counts[gts[j].class_id]++;
    }

    float mAP = 0.0f;
    int valid = 0;
    for (int c = 0; c < g_num_classes; ++c) {
        if (gt_counts[c] == 0) continue;
        float ap = compute_ap(class_pred[c], gt_counts[c]);
        mAP += ap;
        valid++;
    }
    mAP = valid > 0 ? mAP / valid : 0.0f;
    return mAP;
}

static cv::Mat draw_detections(const cv::Mat &image, const std::vector<BBox> &dets) {
    cv::Mat draw = image.clone();
    static const cv::Scalar colors[] = {
        cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0), cv::Scalar(0, 0, 255),
        cv::Scalar(255, 255, 0), cv::Scalar(255, 0, 255), cv::Scalar(0, 255, 255)
    };
    int num_colors = 6;
    for (const auto &b : dets) {
        cv::Scalar color = colors[b.class_id % num_colors];
        cv::rectangle(draw, cv::Point((int)b.x1, (int)b.y1), cv::Point((int)b.x2, (int)b.y2), color, 2);
        std::string label = std::to_string(b.class_id) + ": " + std::to_string((int)(b.conf * 100)) + "%";
        cv::putText(draw, label, cv::Point((int)b.x1, (int)b.y1 - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
    }
    return draw;
}

// ---------- main ----------
int main(int argc, char *argv[]) {
	cv::setNumThreads(8);            // 根据您的 CPU 核数调整
	cv::setUseOptimized(true);       // 启用 SIMD
	
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i+1 < argc) g_model_path = argv[++i];
        else if (arg == "--input" && i+1 < argc) g_input_path = argv[++i];
        else if (arg == "--labels" && i+1 < argc) { g_label_path = argv[++i]; g_compute_map = true; }
        else if (arg == "--output" && i+1 < argc) g_output_path = argv[++i];
        else if (arg == "--conf" && i+1 < argc) g_conf_threshold = atof(argv[++i]);
        else if (arg == "--iou" && i+1 < argc) g_iou_threshold = atof(argv[++i]);
        else if (arg == "--classes" && i+1 < argc) g_num_classes = atoi(argv[++i]);
        else if (arg == "-h" || arg == "--help") {
            printf("Usage: %s --model <mtnn_model> --input <image_dir> [--labels <label_dir>] [--output <output_dir>] [--conf 0.25] [--iou 0.45] [--classes 80]\n", argv[0]);
            return 0;
        }
    }

    if (g_model_path.empty() || g_input_path.empty()) {
        printf("Error: --model and --input are required.\n");
        return -1;
    }

    std::vector<std::string> image_list;
    {
        struct stat st;
        if (stat(g_input_path.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
            image_list = list_images(g_input_path);
        } else {
            cv::Mat test = cv::imread(g_input_path);
            if (!test.empty()) image_list.push_back(g_input_path);
            else {
                FILE *fp = fopen(g_input_path.c_str(), "r");
                if (fp) {
                    char line[512];
                    while (fgets(line, sizeof(line), fp)) {
                        line[strcspn(line, "\n")] = 0;
                        if (strlen(line) > 0) image_list.push_back(line);
                    }
                    fclose(fp);
                }
            }
        }
    }
    if (image_list.empty()) { printf("No images found.\n"); return -1; }

    int num_images = image_list.size();
    std::vector<ImageResult> results(num_images);

    std::vector<std::vector<GTBox>> all_gts;
    if (g_compute_map) {
        all_gts.resize(num_images);
        for (int i = 0; i < num_images; ++i) {
            std::string name = image_list[i].substr(image_list[i].find_last_of("/\\") + 1);
            std::string base = name.substr(0, name.find_last_of('.'));
            std::string label_file = g_label_path + "/" + base + ".txt";
            cv::Mat img = cv::imread(image_list[i]);
            if (!img.empty())
                all_gts[i] = load_labels(label_file, img.cols, img.rows);
        }
    }

    size_t model_size = 0;
    unsigned char *model_data = load_model(g_model_path.c_str(), &model_size);
    if (!model_data) { fprintf(stderr, "Failed to load model\n"); return -1; }

    mtnn_mgr network;
    int status = mtnn_init(&network, model_data, (uint32_t)model_size, NULL);
    free(model_data);
    if (status != 0) { fprintf(stderr, "mtnn_init failed\n"); return -1; }

    mtnn_input_output_num io_num;
    status = mtnn_get(network, MTNN_GET_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (status != 0) { mtnn_destroy(network); return -1; }

    std::vector<mtnn_tensor_mem> input_mem(io_num.n_input);
    status = mtnn_inputs_get(network, io_num.n_input, input_mem.data());
    if (status != 0) { mtnn_destroy(network); return -1; }

    std::vector<mtnn_tensor_attr> out_attrs(io_num.n_output);
    std::vector<mtnn_tensor_mem> out_mems(io_num.n_output);
    for (uint32_t o = 0; o < io_num.n_output; ++o) {
        memset(&out_attrs[o], 0, sizeof(out_attrs[o]));
        out_attrs[o].index = o;
        status = mtnn_get(network, MTNN_GET_OUTPUT_ATTR, &out_attrs[o], sizeof(out_attrs[o]));
        if (status != 0) { mtnn_destroy(network); return -1; }
        memset(&out_mems[o], 0, sizeof(out_mems[o]));
        status = mtnn_outputs_get(network, 1, &out_mems[o]);
        if (status != 0) { mtnn_destroy(network); return -1; }
    }

    double total_time = 0.0;
    for (int i = 0; i < num_images; ++i) {
        bool debug = (i == 0);
        results[i] = infer_image(image_list[i], network, input_mem, out_attrs, out_mems, debug);
        total_time += results[i].inference_time_ms;
        printf("[%d/%d] %s: %.2f ms, %zu detections\n", i+1, num_images,
               image_list[i].c_str(), results[i].inference_time_ms,
               results[i].detections.size());
    }
    printf("Average inference time: %.2f ms\n", total_time / num_images);

    mtnn_destroy(network);

    if (g_compute_map) {
        float map_val = evaluate_map(results, all_gts, g_iou_threshold);
        printf("mAP@%.2f: %.4f\n", g_iou_threshold, map_val);
    }

    if (!g_output_path.empty()) {
        for (int i = 0; i < num_images; ++i) {
            cv::Mat img = cv::imread(image_list[i]);
            if (img.empty()) continue;
            cv::Mat draw = draw_detections(img, results[i].detections);
            std::string out_name = g_output_path + "/" +
                image_list[i].substr(image_list[i].find_last_of("/\\") + 1);
            cv::imwrite(out_name, draw);
        }
        printf("Output images saved to %s\n", g_output_path.c_str());
    }

    return 0;
}
