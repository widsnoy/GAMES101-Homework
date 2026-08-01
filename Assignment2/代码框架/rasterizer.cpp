//
// Created by goksu on 4/6/19.
//

#include "rasterizer.hpp"
#include <algorithm>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <vector>

rst::pos_buf_id
rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions) {
  auto id = get_next_id();
  pos_buf.emplace(id, positions);

  return {id};
}

rst::ind_buf_id
rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices) {
  auto id = get_next_id();
  ind_buf.emplace(id, indices);

  return {id};
}

rst::col_buf_id
rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols) {
  auto id = get_next_id();
  col_buf.emplace(id, cols);

  return {id};
}

auto to_vec4(const Eigen::Vector3f &v3, float w = 1.0f) {
  return Vector4f(v3.x(), v3.y(), v3.z(), w);
}

static bool insideTriangle(float x, float y, const Vector3f *_v) {
  float sign = 0.0f;

  for (int i = 0; i < 3; ++i) {
    const auto &a = _v[i];
    const auto &b = _v[(i + 1) % 3];

    float cross = (x - a.x()) * (b.y() - a.y()) - (y - a.y()) * (b.x() - a.x());

    if (std::abs(cross) > 1e-6f) {
      if (sign == 0.0f) {
        sign = cross;
      } else if (sign * cross < 0.0f) {
        return false;
      }
    }
  }

  return true;
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y,
                                                            const Vector3f *v) {
  float c1 =
      (x * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * y +
       v[1].x() * v[2].y() - v[2].x() * v[1].y()) /
      (v[0].x() * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * v[0].y() +
       v[1].x() * v[2].y() - v[2].x() * v[1].y());
  float c2 =
      (x * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * y +
       v[2].x() * v[0].y() - v[0].x() * v[2].y()) /
      (v[1].x() * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * v[1].y() +
       v[2].x() * v[0].y() - v[0].x() * v[2].y());
  float c3 =
      (x * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * y +
       v[0].x() * v[1].y() - v[1].x() * v[0].y()) /
      (v[2].x() * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * v[2].y() +
       v[0].x() * v[1].y() - v[1].x() * v[0].y());
  return {c1, c2, c3};
}

void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer,
                           col_buf_id col_buffer, Primitive type) {
  auto &buf = pos_buf[pos_buffer.pos_id];
  auto &ind = ind_buf[ind_buffer.ind_id];
  auto &col = col_buf[col_buffer.col_id];

  float f1 = (50 - 0.1) / 2.0;
  float f2 = (50 + 0.1) / 2.0;

  Eigen::Matrix4f mvp = projection * view * model;
  for (auto &i : ind) {
    Triangle t;
    Eigen::Vector4f v[] = {mvp * to_vec4(buf[i[0]], 1.0f),
                           mvp * to_vec4(buf[i[1]], 1.0f),
                           mvp * to_vec4(buf[i[2]], 1.0f)};
    // Homogeneous division
    for (auto &vec : v) {
      vec /= vec.w();
    }
    // Viewport transformation
    for (auto &vert : v) {
      vert.x() = 0.5 * width * (vert.x() + 1.0);
      vert.y() = 0.5 * height * (vert.y() + 1.0);
      vert.z() = vert.z() * f1 + f2;
    }

    for (int i = 0; i < 3; ++i) {
      t.setVertex(i, v[i].head<3>());
    }

    auto col_x = col[i[0]];
    auto col_y = col[i[1]];
    auto col_z = col[i[2]];

    t.setColor(0, col_x[0], col_x[1], col_x[2]);
    t.setColor(1, col_y[0], col_y[1], col_y[2]);
    t.setColor(2, col_z[0], col_z[1], col_z[2]);

    rasterize_triangle(t);
  }
}

// Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle &t) {
  auto v = t.toVector4();

  // TODO : Find out the bounding box of current triangle.
  // iterate through the pixel and find if the current pixel is inside the
  // triangle
  float min_x = std::ceil(std::min({v[0].x(), v[1].x(), v[2].x()}));
  float max_x = std::floor(std::max({v[0].x(), v[1].x(), v[2].x()}));
  float min_y = std::ceil(std::min({v[0].y(), v[1].y(), v[2].y()}));
  float max_y = std::floor(std::max({v[0].y(), v[1].y(), v[2].y()}));

  float slen = 1.0 / super_sampler_1d;

  for (float ix = min_x; ix <= max_x; ix++) {
    for (float iy = min_y; iy <= max_y; iy++) {
      int k = 0;
      for (int ii = 0; ii < super_sampler_1d; ii++) {
        for (int jj = 0; jj < super_sampler_1d; jj++) {
          k++;
          float x = ix + slen / 2.0 + ii * slen;
          float y = iy + slen / 2.0 + jj * slen;
          if (!insideTriangle(x, y, t.v)) {
            continue;
          }
          // If so, use the following code to get the interpolated z value.
          auto [alpha, beta, gamma] = computeBarycentric2D(x, y, t.v);
          float w_reciprocal =
              1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
          float z_interpolated = alpha * v[0].z() / v[0].w() +
                                 beta * v[1].z() / v[1].w() +
                                 gamma * v[2].z() / v[2].w();
          z_interpolated *= w_reciprocal;

          // TODO : set the current pixel (use the set_pixel function) to the
          // color of the triangle (use getColor function) if it should be
          // painted.
          int ind = this->get_super_sampled_index(x, y, k - 1);
          if (z_interpolated > this->depth_buf[ind]) {
            this->set_pixel(Eigen::Vector3f(x, y, k - 1), t.getColor());
            this->depth_buf[ind] = z_interpolated;
          }
        }
      }
    }
  }

  for (int i = 0; i < frame_buf.size();
       i += super_sampler_1d * super_sampler_1d) {
    Vector3f avg(0, 0, 0);
    for (int j = 0; j < super_sampler_1d * super_sampler_1d; j++) {
      avg += frame_buf[i + j];
    }
    avg /= super_sampler_1d * super_sampler_1d;
    frame_buf_resolved[i / (super_sampler_1d * super_sampler_1d)] = avg;
  }
}

void rst::rasterizer::set_model(const Eigen::Matrix4f &m) { model = m; }

void rst::rasterizer::set_view(const Eigen::Matrix4f &v) { view = v; }

void rst::rasterizer::set_projection(const Eigen::Matrix4f &p) {
  projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff) {
  if ((buff & rst::Buffers::Color) == rst::Buffers::Color) {
    std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
  }
  if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth) {
    std::fill(depth_buf.begin(), depth_buf.end(),
              -std::numeric_limits<float>::infinity());
  }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h) {
  frame_buf.resize(w * h * super_sampler_1d * super_sampler_1d);
  depth_buf.resize(w * h * super_sampler_1d * super_sampler_1d);
  frame_buf_resolved.resize(w * h);
}

int rst::rasterizer::get_index(int x, int y) {
  return (height - 1 - y) * width + x;
}

int rst::rasterizer::get_super_sampled_index(int x, int y, int k) {
  return get_index(x, y) * super_sampler_1d * super_sampler_1d + k;
}

void rst::rasterizer::set_pixel(const Eigen::Vector3f &point,
                                const Eigen::Vector3f &color) {
  // old index: auto ind = point.y() + point.x() * width;
  auto ind = get_super_sampled_index(point.x(), point.y(), point.z());
  frame_buf[ind] = color;
}
