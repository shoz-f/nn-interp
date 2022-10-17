/***  File Header  ************************************************************/
/**
* nonmaxsuppression.cpp
*
* Tiny ML post processing libraies: NMS
* @author      Shozo Fukuda
* @date create Tue Jul 13 14:25:06 JST 2021
* System       Windows10, WSL2/Ubuntu20.04.2, Linux Mint<br>
*
**/
/**************************************************************************{{{*/

#include "tiny_ml.h"
#include "postprocess.h"

#include <queue>

/***  Class Header  *******************************************************}}}*/
/**
* bounding box
* @par DESCRIPTION
*   it holds bbox and score needed for NMS and provides IOU function.
**/
/**************************************************************************{{{*/
class Box {
//LIFECYCLE:
public:
    Box(const float box[4], float score, unsigned int box_repr=0) {
        switch (box_repr) {
        case 2:
            mBBox[0] = box[0];
            mBBox[1] = box[1];
            mBBox[2] = box[2];
            mBBox[3] = box[3];
            mArea = (box[2]-box[0]+1)*(box[3]-box[1]+1);
            break;

        case 1:
            mBBox[0] = box[0];
            mBBox[1] = box[1];
            mBBox[2] = box[0] + box[2];
            mBBox[3] = box[1] + box[3];
            mArea = box[2]*box[3];
            break;

        case 0:
        default:
            mBBox[0] = static_cast<float>(box[0] - box[2]/2.0);
            mBBox[1] = static_cast<float>(box[1] - box[3]/2.0);
            mBBox[2] = static_cast<float>(box[0] + box[2]/2.0);
            mBBox[3] = static_cast<float>(box[1] + box[3]/2.0);
            mArea = box[2]*box[3];
            break;
        }
        
        mScore = score;
    }

//ACTION:
public:
    // calc Intersection over Union
    float iou(const Box& x) const {
        float x1 = std::max(mBBox[0], x.mBBox[0]);
        float y1 = std::max(mBBox[1], x.mBBox[1]);
        float x2 = std::min(mBBox[2], x.mBBox[2]);
        float y2 = std::min(mBBox[3], x.mBBox[3]);
        
        if (x1 < x2 && y1 < y2) {
            float v_intersection = (x2 - x1)*(y2 - y1);
            float v_union        = mArea + x.mArea - v_intersection;
            return v_intersection/v_union;
        }
        else {
            return 0.0;
        }
    }

    // Comparison operation
    bool less(const Box& b) const {
        return mScore < b.mScore;
    }

    // put out the scaled BBox in JSON formatting
    json to_json() const {
        auto result = json::array();
        result.push_back(mScore);
        result.push_back(mBBox[0]);
        result.push_back(mBBox[1]);
        result.push_back(mBBox[2]);
        result.push_back(mBBox[3]);
        return result;
    }

//ACCESSOR:
public:
    void set_score(float score) {
        mScore = score;
    }
    
    float get_score() {
        return mScore;
    }

//ATTRIBUTE:
protected:
    float         mBBox[4];
    float         mArea;
    float         mScore;
};

// Comparison operator for Boxes
bool operator< (const Box& a, const Box& b) {
    return a.less(b);
}

/***  Module Header  ******************************************************}}}*/
/**
* Non Maximum Suppression for Multi Class
* @par DESCRIPTION
*   run non-maximum on every class
*
* @retval json
**/
/**************************************************************************{{{*/
std::string
non_max_suppression_multi_class(
unsigned int num_boxes,
unsigned int box_repr,
const float* boxes,
unsigned int num_class,
const float* scores,
float         iou_threshold,
float         score_threshold,
float         sigma)
{
    json res;
    std::priority_queue<Box> candidates;

    // run nms over each classification class.
    for (unsigned int class_id = 0; class_id < num_class; class_id++) {
        // pick up candidates for focus class
        const float* _boxes  = boxes;
        const float* _scores = scores;
        for (unsigned int i = 0; i < num_boxes; i++, _boxes += 4, _scores += num_class) {
            if (_scores[class_id] > score_threshold) {
                candidates.emplace(_boxes, _scores[class_id], box_repr);
            }
        }
        if (candidates.empty()) continue;

        // perform iou filtering
        std::string class_name = gSys.label(class_id);
        do {
            Box selected = candidates.top();  candidates.pop();

            res[class_name].push_back(selected.to_json());

            while (!candidates.empty()) {
                float iou = selected.iou(candidates.top());
                if (iou < iou_threshold) { break; }

                if (sigma > 0.0) {
                    Box next = candidates.top(); candidates.pop();
                    float soft_nms_score = next.get_score()*exp(-(iou*iou)/sigma);
                    if (soft_nms_score > score_threshold) {
                        next.set_score(soft_nms_score);
                        candidates.push(next);
                    }
                }
                else {
                    candidates.pop();
                }
            }
        } while (!candidates.empty());
    }

    return res.dump();
}

/***  Module Header  ******************************************************}}}*/
/**
* Non Maximum Suppression for Multi Class
* @par DESCRIPTION
*   run non-maximum on every class
*
* @retval json
**/
/**************************************************************************{{{*/
std::string
non_max_suppression_multi_class(SysInfo&, const void* args)
{
    PACK(
    struct Prms {
        unsigned int num_boxes;
        unsigned int box_repr;
        unsigned int num_class;
        float         iou_threshold;
        float         score_threshold;
        float         sigma;
        float         table[1];
    });
    const Prms*  prms = reinterpret_cast<const Prms*>(args);

    return non_max_suppression_multi_class(
        prms->num_boxes,
        prms->box_repr,
        &prms->table[0],
        prms->num_class,
        &prms->table[4*prms->num_boxes],
        prms->iou_threshold,
        prms->score_threshold,
        prms->sigma
    );
}

/*** nonmaxsuppression.cpp ************************************************}}}*/
