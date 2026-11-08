#include "trail_view.hpp"

#include <imgui_impl_opengl3_loader.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

namespace wildlife {
namespace {

struct Vertex {
  ImVec2 position;
  ImU32 color;
  float edge;
};
struct DrawArgs {
  ImVec2 origin, size, clip_min, clip_max;
  std::size_t view;
};
struct Batch {
  using DrawArrays = void(APIENTRY *)(GLenum, GLint, GLsizei);
  std::vector<std::vector<Vertex>> views;
  std::size_t next_view = 0;
  int frame = -1;
  GLuint program = 0, vao = 0, buffer = 0;
  GLint display = -1;
  DrawArrays draw_arrays = nullptr;
  bool attempted = false;
} batch;

GLuint shader(GLenum type, const char *source) {
  const GLuint id = glCreateShader(type);
  glShaderSource(id, 1, &source, nullptr);
  glCompileShader(id);
  GLint ok = 0;
  glGetShaderiv(id, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    char message[1024]{};
    glGetShaderInfoLog(id, sizeof(message), nullptr, message);
    std::fprintf(stderr, "Trail shader: %s\n", message);
    glDeleteShader(id);
    return 0;
  }
  return id;
}

bool initialize() {
  if (batch.attempted)
    return batch.program != 0;
  batch.attempted = true;
  batch.draw_arrays =
      reinterpret_cast<Batch::DrawArrays>(glfwGetProcAddress("glDrawArrays"));
  if (!batch.draw_arrays)
    return false;
  const GLuint vertex = shader(GL_VERTEX_SHADER, R"glsl(#version 150
in vec2 position;
in vec4 color;
in float edge;
uniform mat4 viewportData;
out vec4 tint;
out float across;
void main() {
  vec2 p = (position - viewportData[0].xy) / viewportData[0].zw;
  gl_Position = vec4(p.x * 2.0 - 1.0, 1.0 - p.y * 2.0, 0.0, 1.0);
  tint = color;
  across = edge;
}
)glsl");
  const GLuint fragment = shader(GL_FRAGMENT_SHADER, R"glsl(#version 150
in vec4 tint;
in float across;
out vec4 result;
void main() {
  float core = exp(-across * across * 85.0);
  float halo = exp(-across * across * 5.0);
  float coverage = (0.85 * core + 0.32 * halo)
                 * (1.0 - smoothstep(0.85, 1.0, abs(across)));
  result = vec4(mix(tint.rgb, vec3(1.0), core * 0.16),
                tint.a * min(coverage, 1.0));
}
)glsl");
  if (!vertex || !fragment) {
    if (vertex)
      glDeleteShader(vertex);
    if (fragment)
      glDeleteShader(fragment);
    return false;
  }
  const GLuint program = glCreateProgram();
  glAttachShader(program, vertex);
  glAttachShader(program, fragment);
  glLinkProgram(program);
  glDeleteShader(vertex);
  glDeleteShader(fragment);
  GLint ok = 0;
  glGetProgramiv(program, GL_LINK_STATUS, &ok);
  if (!ok) {
    char message[1024]{};
    glGetProgramInfoLog(program, sizeof(message), nullptr, message);
    std::fprintf(stderr, "Trail program: %s\n", message);
    glDeleteProgram(program);
    return false;
  }
  batch.program = program;
  batch.display = glGetUniformLocation(program, "viewportData");
  glGenVertexArrays(1, &batch.vao);
  glGenBuffers(1, &batch.buffer);
  glBindVertexArray(batch.vao);
  glBindBuffer(GL_ARRAY_BUFFER, batch.buffer);
  const auto position = GLuint(glGetAttribLocation(program, "position"));
  const auto color = GLuint(glGetAttribLocation(program, "color"));
  const auto edge = GLuint(glGetAttribLocation(program, "edge"));
  glEnableVertexAttribArray(position);
  glEnableVertexAttribArray(color);
  glEnableVertexAttribArray(edge);
  glVertexAttribPointer(position, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                        reinterpret_cast<void *>(offsetof(Vertex, position)));
  glVertexAttribPointer(color, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(Vertex),
                        reinterpret_cast<void *>(offsetof(Vertex, color)));
  glVertexAttribPointer(edge, 1, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                        reinterpret_cast<void *>(offsetof(Vertex, edge)));
  return true;
}

void render(const ImDrawList *, const ImDrawCmd *command) {
  const auto &args = *static_cast<const DrawArgs *>(command->UserCallbackData);
  const auto &vertices = batch.views[args.view];
  if (vertices.empty() || !initialize())
    return;
  GLint viewport[4]{};
  glGetIntegerv(GL_VIEWPORT, viewport);
  const float sx = viewport[2] / args.size.x, sy = viewport[3] / args.size.y;
  const int left = std::clamp(
      int(std::floor((args.clip_min.x - args.origin.x) * sx)), 0, viewport[2]);
  const int right = std::clamp(
      int(std::ceil((args.clip_max.x - args.origin.x) * sx)), 0, viewport[2]);
  const int top = std::clamp(
      int(std::floor((args.clip_min.y - args.origin.y) * sy)), 0, viewport[3]);
  const int bottom = std::clamp(
      int(std::ceil((args.clip_max.y - args.origin.y) * sy)), 0, viewport[3]);
  if (left >= right || top >= bottom)
    return;
  glEnable(GL_SCISSOR_TEST);
  glScissor(viewport[0] + left, viewport[1] + viewport[3] - bottom,
            right - left, bottom - top);
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_CULL_FACE);
  glEnable(GL_BLEND);
  glBlendEquation(GL_FUNC_ADD);
  glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE,
                      GL_ONE_MINUS_SRC_ALPHA);
  glUseProgram(batch.program);
  const std::array<float, 16> layout{args.origin.x, args.origin.y, args.size.x,
                                     args.size.y};
  glUniformMatrix4fv(batch.display, 1, GL_FALSE, layout.data());
  glBindVertexArray(batch.vao);
  glBindBuffer(GL_ARRAY_BUFFER, batch.buffer);
  glBufferData(GL_ARRAY_BUFFER, GLsizeiptr(vertices.size() * sizeof(Vertex)),
               vertices.data(), GL_STREAM_DRAW);
  constexpr GLenum triangle_strip = 0x0005;
  batch.draw_arrays(triangle_strip, 0, GLsizei(vertices.size()));
}

ImVec2 normal(ImVec2 a, ImVec2 b) {
  const float dx = b.x - a.x, dy = b.y - a.y;
  const float length = std::hypot(dx, dy);
  return length > 1e-4f ? ImVec2{-dy / length, dx / length} : ImVec2{0, 1};
}

} // namespace

bool initialize_glow_trails() { return initialize(); }

void begin_glow_trails(ImDrawList &draw) {
  if (batch.frame != ImGui::GetFrameCount()) {
    batch.frame = ImGui::GetFrameCount();
    batch.next_view = 0;
  }
  const auto index = batch.next_view++;
  if (index == batch.views.size())
    batch.views.emplace_back();
  batch.views[index].clear();
  const auto *viewport = ImGui::GetMainViewport();
  DrawArgs args{viewport->Pos, viewport->Size, draw.GetClipRectMin(),
                draw.GetClipRectMax(), index};
  draw.AddCallback(render, &args, sizeof(args));
  draw.AddCallback(ImDrawCallback_ResetRenderState, nullptr);
}

void add_glow_trail(std::span<const GlowTrailPoint> points, ImU32 color,
                    float half_width) {
  if (points.size() < 2)
    return;
  auto &vertices = batch.views[batch.next_view - 1];
  // Degenerate triangles separate paths within a single draw call. A path
  // contributes only two vertices per point, including at its joined corners.
  const bool separate = !vertices.empty();
  if (separate)
    vertices.push_back(vertices.back());
  for (std::size_t i = 0; i < points.size(); ++i) {
    const auto p = points[i].position;
    const auto before =
        normal(points[i ? i - 1 : i].position, points[i ? i : 1].position);
    const auto after =
        normal(points[i + 1 < points.size() ? i : i - 1].position,
               points[std::min(i + 1, points.size() - 1)].position);
    float nx = before.x + after.x, ny = before.y + after.y;
    const float length = std::hypot(nx, ny);
    if (length < 1e-3f) {
      nx = after.x;
      ny = after.y;
    } else {
      nx /= length;
      ny /= length;
    }
    const float scale = half_width / std::max(.5f, nx * after.x + ny * after.y);
    const auto alpha = ImU32(std::clamp(points[i].opacity, 0.f, 1.f) * 255.f);
    const ImU32 tint = (color & ~IM_COL32_A_MASK) | (alpha << IM_COL32_A_SHIFT);
    const Vertex a{{p.x + nx * scale, p.y + ny * scale}, tint, 1};
    if (i == 0 && separate)
      vertices.push_back(a);
    vertices.push_back(a);
    vertices.push_back({{p.x - nx * scale, p.y - ny * scale}, tint, -1});
  }
}

void shutdown_glow_trails() {
  if (batch.buffer)
    glDeleteBuffers(1, &batch.buffer);
  if (batch.vao)
    glDeleteVertexArrays(1, &batch.vao);
  if (batch.program)
    glDeleteProgram(batch.program);
  batch = {};
}

} // namespace wildlife
