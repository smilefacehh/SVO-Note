// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// SVO is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// SVO is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <svo/config.h>
#include <svo/frame.h>
#include <svo/point.h>
#include <svo/feature.h>
#include <svo/initialization.h>
#include <svo/feature_detection.h>
#include <vikit/math_utils.h>
#include <vikit/homography.h>

namespace svo {
namespace initialization {

/**
 * 第一帧提取Fast特征点
*/
InitResult KltHomographyInit::addFirstFrame(FramePtr frame_ref)
{
  reset();
  /**
   * Fast特征提取
   * 1、Fast特征点提取，划分网格，每个网格只存得分最高的一个特征点，最后根据阈值选择得分较高的特征点集合
   * 2、存特征点的像素坐标、单位球坐标
  */
  detectFeatures(frame_ref, px_ref_, f_ref_);
  // 第一帧特征点希望多一些，初始化精度很重要
  if(px_ref_.size() < 100)
  {
    SVO_WARN_STREAM_THROTTLE(2.0, "First image has less than 100 features. Retry in more textured environment.");
    return FAILURE;
  }
  // 设为参考帧
  frame_ref_ = frame_ref;
  px_cur_.insert(px_cur_.begin(), px_ref_.begin(), px_ref_.end());
  return SUCCESS;
}

/**
 * 第二帧与第一帧，计算H恢复R、t，三角化点
 * 要求两帧匹配点够多，平移距离够大
 * 1、LK跟踪两帧特征点，计算当前帧特征点像素坐标px_cur、单位球坐标f_cur，匹配点像素位置差disparities
 * 2、两帧匹配点计算单应矩阵H，恢复位姿T_cur_from_ref_，三角化xyz_in_cur_
 * 3、设定尺度
 * 4、图像帧添加特征点，特征点添加观测帧
*/
InitResult KltHomographyInit::addSecondFrame(FramePtr frame_cur)
{
  // LK跟踪两帧特征点，计算当前帧特征点像素坐标px_cur、单位球坐标f_cur，匹配点像素位置差disparities
  trackKlt(frame_ref_, frame_cur, px_ref_, px_cur_, f_ref_, f_cur_, disparities_);
  SVO_INFO_STREAM("Init: KLT tracked "<< disparities_.size() <<" features");

  // 匹配点太少，等待下一帧重新计算
  if(disparities_.size() < Config::initMinTracked())
    return FAILURE;

  // 像素位置误差较小，不是关键帧，等待下一帧重新计算
  double disparity = vk::getMedian(disparities_);
  SVO_INFO_STREAM("Init: KLT "<<disparity<<"px average disparity.");
  if(disparity < Config::initMinDisparity())
    return NO_KEYFRAME;

  // 两帧匹配点计算单应矩阵H，恢复位姿T_cur_from_ref_，三角化xyz_in_cur_
  computeHomography(
      f_ref_, f_cur_,
      frame_ref_->cam_->errorMultiplier2(), Config::poseOptimThresh(),
      inliers_, xyz_in_cur_, T_cur_from_ref_);
  SVO_INFO_STREAM("Init: Homography RANSAC "<<inliers_.size()<<" inliers.");

  if(inliers_.size() < Config::initMinInliers())
  {
    SVO_WARN_STREAM("Init WARNING: "<<Config::initMinInliers()<<" inliers minimum required.");
    return FAILURE;
  }

  // Rescale the map such that the mean scene depth is equal to the specified scale
  vector<double> depth_vec;
  for(size_t i=0; i<xyz_in_cur_.size(); ++i)
    depth_vec.push_back((xyz_in_cur_[i]).z());
  // 深度中值
  double scene_depth_median = vk::getMedian(depth_vec);
  /**
   * 尺度不确定性怎么来的？
   * 两帧之间通过对极约束计算R、t，这个t是个单位向量，任意缩放对极约束都成立，因为我们并不知道点的实际深度。
   * 再根据R、t进行三角化，如果t确定了，三角化点也是确定的。后续帧，用三角化的点进行pnp求解位姿，得到的R、t与前面的R、t在同一个尺度下了。
   * 所以要指定t的尺度，或者指定第一帧三角化点的深度值（通常设为1），整个系统的尺度就固定为1了。
  */
  // 单目尺度不确定，人为设置一个尺度为mapScale（通常为1）
  // 深度值缩放1/f，等价于平移缩放1/f，因为是三角形的两个边，等比例缩放
  double scale = Config::mapScale()/scene_depth_median;

  // 计算当前帧位姿，旋转不用考虑尺度
  frame_cur->T_f_w_ = T_cur_from_ref_ * frame_ref_->T_f_w_;
  // 平移尺度缩放一下
  frame_cur->T_f_w_.translation() =
      -frame_cur->T_f_w_.rotation_matrix()*(frame_ref_->pos() + scale*(frame_cur->pos() - frame_ref_->pos()));

  // For each inlier create 3D point and add feature in both frames
  SE3 T_world_cur = frame_cur->T_f_w_.inverse();
  for(vector<int>::iterator it=inliers_.begin(); it!=inliers_.end(); ++it)
  {
    Vector2d px_cur(px_cur_[*it].x, px_cur_[*it].y);
    Vector2d px_ref(px_ref_[*it].x, px_ref_[*it].y);
    if(frame_ref_->cam_->isInFrame(px_cur.cast<int>(), 10) && frame_ref_->cam_->isInFrame(px_ref.cast<int>(), 10) && xyz_in_cur_[*it].z() > 0)
    {
      Vector3d pos = T_world_cur * (xyz_in_cur_[*it]*scale);
      Point* new_point = new Point(pos);

      // 图像帧添加特征点，特征点添加观测帧
      Feature* ftr_cur(new Feature(frame_cur.get(), new_point, px_cur, f_cur_[*it], 0));
      frame_cur->addFeature(ftr_cur);
      new_point->addFrameRef(ftr_cur);

      // 图像帧添加特征点，特征点添加观测帧
      Feature* ftr_ref(new Feature(frame_ref_.get(), new_point, px_ref, f_ref_[*it], 0));
      frame_ref_->addFeature(ftr_ref);
      new_point->addFrameRef(ftr_ref);
    }
  }
  return SUCCESS;
}

void KltHomographyInit::reset()
{
  px_cur_.clear();
  frame_ref_.reset();
}

/**
 * Fast特征提取
 * 1、Fast特征点提取，划分网格，每个网格只存得分最高的一个特征点，最后根据阈值选择得分较高的特征点集合
 * 2、存特征点的像素坐标、单位球坐标
*/
void detectFeatures(
    FramePtr frame,
    vector<cv::Point2f>& px_vec,
    vector<Vector3d>& f_vec)
{
  // Fast特征点提取，划分网格，每个网格只存得分最高的一个特征点，最后根据阈值选择得分较高的特征点集合；存new_features
  Features new_features;
  feature_detection::FastDetector detector(
      frame->img().cols, frame->img().rows, Config::gridSize(), Config::nPyrLevels());
  detector.detect(frame.get(), frame->img_pyr_, Config::triangMinCornerScore(), new_features);

  // now for all maximum corners, initialize a new seed
  // 存特征点的像素坐标、单位球坐标
  px_vec.clear(); px_vec.reserve(new_features.size());
  f_vec.clear(); f_vec.reserve(new_features.size());
  std::for_each(new_features.begin(), new_features.end(), [&](Feature* ftr){
    px_vec.push_back(cv::Point2f(ftr->px[0], ftr->px[1]));
    f_vec.push_back(ftr->f);
    delete ftr;
  });
}

/**
 * LK跟踪两帧特征点，计算当前帧特征点像素坐标px_cur、单位球坐标f_cur，匹配点像素位置差disparities
 * 删除outlier点
*/
void trackKlt(
    FramePtr frame_ref,
    FramePtr frame_cur,
    vector<cv::Point2f>& px_ref,
    vector<cv::Point2f>& px_cur,
    vector<Vector3d>& f_ref,
    vector<Vector3d>& f_cur,
    vector<double>& disparities)
{
  const double klt_win_size = 30.0;
  const int klt_max_iter = 30;
  const double klt_eps = 0.001;
  vector<uchar> status;
  vector<float> error;
  vector<float> min_eig_vec;
  cv::TermCriteria termcrit(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, klt_max_iter, klt_eps);
  // LK光流跟踪两帧的特征点，第二帧特征点用第一帧特征点初始化，如果有IMU，可以用速度预测得到一个更准确的初始位置
  cv::calcOpticalFlowPyrLK(frame_ref->img_pyr_[0], frame_cur->img_pyr_[0],
                           px_ref, px_cur,
                           status, error,
                           cv::Size2i(klt_win_size, klt_win_size),
                           4, termcrit, cv::OPTFLOW_USE_INITIAL_FLOW);

  vector<cv::Point2f>::iterator px_ref_it = px_ref.begin();
  vector<cv::Point2f>::iterator px_cur_it = px_cur.begin();
  vector<Vector3d>::iterator f_ref_it = f_ref.begin();
  f_cur.clear(); f_cur.reserve(px_cur.size());
  disparities.clear(); disparities.reserve(px_cur.size());
  for(size_t i=0; px_ref_it != px_ref.end(); ++i)
  {
    // 删除未匹配上的点
    if(!status[i])
    {
      px_ref_it = px_ref.erase(px_ref_it);
      px_cur_it = px_cur.erase(px_cur_it);
      f_ref_it = f_ref.erase(f_ref_it);
      continue;
    }
    // 单位球坐标
    f_cur.push_back(frame_cur->c2f(px_cur_it->x, px_cur_it->y));
    // 像素点位置差
    disparities.push_back(Vector2d(px_ref_it->x - px_cur_it->x, px_ref_it->y - px_cur_it->y).norm());
    ++px_ref_it;
    ++px_cur_it;
    ++f_ref_it;
  }
}

/**
 * 两帧匹配点计算单应矩阵H，恢复位姿T，三角化
 * 注：恢复H，要求点在同一平面上
*/
void computeHomography(
    const vector<Vector3d>& f_ref,
    const vector<Vector3d>& f_cur,
    double focal_length,
    double reprojection_threshold,
    vector<int>& inliers,
    vector<Vector3d>& xyz_in_cur,
    SE3& T_cur_from_ref)
{
  vector<Vector2d > uv_ref(f_ref.size());
  vector<Vector2d > uv_cur(f_cur.size());
  for(size_t i=0, i_max=f_ref.size(); i<i_max; ++i)
  {
    uv_ref[i] = vk::project2d(f_ref[i]);
    uv_cur[i] = vk::project2d(f_cur[i]);
  }
  vk::Homography Homography(uv_ref, uv_cur, focal_length, reprojection_threshold);
  Homography.computeSE3fromMatches();
  vector<int> outliers;
  vk::computeInliers(f_cur, f_ref,
                     Homography.T_c2_from_c1.rotation_matrix(), Homography.T_c2_from_c1.translation(),
                     reprojection_threshold, focal_length,
                     xyz_in_cur, inliers, outliers);
  T_cur_from_ref = Homography.T_c2_from_c1;
}


} // namespace initialization
} // namespace svo
