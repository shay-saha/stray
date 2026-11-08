#include "app.hpp"
#include "earth_texture.hpp"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#if defined(__APPLE__) && !defined(GL_SILENCE_DEPRECATION)
#define GL_SILENCE_DEPRECATION
#endif
#include <GLFW/glfw3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <curl/curl.h>
#include <fstream>
#include <numbers>
#include <string_view>
#include <thread>
#include <vector>

namespace {

enum class RunMode {
  Interactive,
  GlobeSmoke,
  DotsSmoke,
  ZoomSmoke,
  ForecastSmoke
};

void glfw_error(int code, const char *description) {
  std::fprintf(stderr, "GLFW error %d: %s\n", code, description);
}

GLFWwindow *create_window() {
  int width = 1440;
  int height = 900;
  if (GLFWmonitor *monitor = glfwGetPrimaryMonitor()) {
    int work_width = 0;
    int work_height = 0;
    glfwGetMonitorWorkarea(monitor, nullptr, nullptr, &work_width,
                           &work_height);
    if (work_width > 0 && work_height > 0) {
      width = std::min(width, work_width * 92 / 100);
      height = std::min(height, work_height * 92 / 100);
    }
  }
  GLFWwindow *window =
      glfwCreateWindow(width, height, "Wildlife Atlas", nullptr, nullptr);
  if (window) {
    glfwSetWindowSizeLimits(window, std::min(width, 760), std::min(height, 480),
                            GLFW_DONT_CARE, GLFW_DONT_CARE);
  }
  return window;
}

int render_loop(GLFWwindow *window, RunMode mode, const char *executable,
                std::string_view test_species) {
  constexpr int warmup_frames = 30;
  constexpr int measured_frames = 240;
  const bool smoke = mode != RunMode::Interactive;
  if (mode == RunMode::ZoomSmoke) {
    for (const auto location :
         {ImVec2{45.9f, 7.6f}, ImVec2{0, 179.9f}, ImVec2{86, -35}}) {
      const double lat = location.x + .00001, lon = location.y + .00001;
      GlobeView camera({500, 400}, 50000.f, location.x, location.y, 28);
      ImVec2 screen;
      double back_lat = 0, back_lon = 0;
      if (!camera.project(lat, lon, screen) ||
          !camera.unproject(screen, back_lat, back_lon) ||
          std::hypot(lat - back_lat,
                     (lon - back_lon) *
                         std::cos(lat * std::numbers::pi / 180)) *
                  111000 >
              2) {
        std::fprintf(stderr,
                     "Regional terrain projection failed its round-trip: "
                     "%.9f,%.9f -> %.9f,%.9f.\n",
                     lat, lon, back_lat, back_lon);
        return 1;
      }
      GlobeView after({500, 400}, 60000.f, location.x, location.y, 28);
      double centre_lat = 0, centre_lon = 0;
      if (!after.center_for_anchor(screen, lat, lon, centre_lat, centre_lon))
        return 1;
      GlobeView anchored({500, 400}, 60000.f, centre_lat, centre_lon, 28);
      ImVec2 anchored_screen;
      if (!anchored.project(lat, lon, anchored_screen) ||
          std::hypot(anchored_screen.x - screen.x,
                     anchored_screen.y - screen.y) > 2.f) {
        std::fputs("Wheel zoom failed to preserve its terrain anchor.\n",
                   stderr);
        return 1;
      }
    }
    std::puts("Regional camera round-trip and zoom-anchor checks passed.");
  }
  wildlife::App app(executable);
  if (const char *forecast = std::getenv("ATLAS_FORECAST")) {
    if (!app.load_forecast(forecast))
      return 1;
  }
  const char *focus_env = std::getenv("ATLAS_FOCUS_SPECIES");
  const std::string focus_name =
      focus_env ? focus_env : std::string(test_species);
  bool focus_done =
      mode != RunMode::DotsSmoke && mode != RunMode::ZoomSmoke && !focus_env;
  const auto configure_camera = [&] {
    if (!focus_done) {
      if (!app.focus_dots(focus_name))
        return false;
      focus_done = true;
    }
    if (const char *zoom = std::getenv("ATLAS_TEST_ZOOM"))
      app.test_zoom(std::strtof(zoom, nullptr));
    if (const char *location = std::getenv("ATLAS_TEST_LOCATION")) {
      double lat = 0, lon = 0;
      if (std::sscanf(location, "%lf,%lf", &lat, &lon) != 2)
        return false;
      const char *zoom = std::getenv("ATLAS_TEST_ZOOM");
      app.focus_location(lat, lon,
                         zoom ? std::strtof(zoom, nullptr)
                              : wildlife::App::max_zoom);
    }
    return true;
  };
  if ((focus_done || app.model_ready()) && !configure_camera())
    return 2;
  if (mode == RunMode::ForecastSmoke && !std::getenv("ATLAS_FORECAST")) {
    std::fputs("Set ATLAS_FORECAST to a real WildCast JSON file for forecast "
               "smoke testing.\n",
               stderr);
    return 1;
  }
  double measure_start = 0.0;
  const double startup = glfwGetTime();
  const double stream_warmup =
      std::getenv("ATLAS_STREAM_WARMUP")
          ? std::max(0., std::atof(std::getenv("ATLAS_STREAM_WARMUP")))
          : 0;
  int frames = 0;
  int peak_vertices = 0;
  bool forecast_day_set = false;
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    if (glfwGetWindowAttrib(window, GLFW_ICONIFIED) || width == 0 ||
        height == 0) {
      if (smoke) {
        std::fputs("Smoke test lost its framebuffer.\n", stderr);
        return 1;
      }
      glfwWaitEventsTimeout(0.1);
      continue;
    }

    if (smoke && frames == warmup_frames) {
      glFinish();
      measure_start = glfwGetTime();
    }

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    if (!forecast_day_set && app.model_ready()) {
      if (const char *day = std::getenv("ATLAS_TEST_FORECAST_DAY"))
        app.test_forecast_day(std::atof(day),
                              std::getenv("ATLAS_TEST_AUTOPLAY"));
      forecast_day_set = true;
    }
    if (!focus_done && app.model_ready() && !configure_camera())
      return 2;
    if (mode == RunMode::ZoomSmoke) {
      const float t = std::clamp(
          float(frames) / (warmup_frames + measured_frames - 1), 0.f, 1.f);
      app.test_zoom(std::exp(std::log(wildlife::App::max_zoom) *
                             (1.f - std::abs(2.f * t - 1.f))));
    }
    app.draw();
    ImGui::Render();
    glViewport(0, 0, width, height);
    glClearColor(0.035f, 0.043f, 0.055f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImDrawData *draw_data = ImGui::GetDrawData();
    ImGui_ImplOpenGL3_RenderDrawData(draw_data);
    if (smoke && (!focus_done || glfwGetTime() - startup < stream_warmup)) {
      if (glfwGetTime() - startup > 180) {
        std::fputs("No model-backed observations became available.\n", stderr);
        return 1;
      }
      glfwSwapBuffers(window);
      frames = 0;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }
    if (smoke && frames == warmup_frames + measured_frames - 1) {
      if (const char *path = std::getenv("ATLAS_SCREENSHOT")) {
        std::vector<unsigned char> pixels(width * height * 3);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE,
                     pixels.data());
        std::ofstream image(path, std::ios::binary);
        image << "P6\n" << width << ' ' << height << "\n255\n";
        for (int y = height - 1; y >= 0; --y)
          image.write(
              reinterpret_cast<const char *>(pixels.data() + y * width * 3),
              width * 3);
        if (!image)
          return 1;
      }
    }
    glfwSwapBuffers(window);

    if (smoke && frames >= warmup_frames) {
      if (draw_data->TotalVtxCount <= 0) {
        std::fputs("Smoke test produced no ImGui geometry.\n", stderr);
        return 1;
      }
      peak_vertices = std::max(peak_vertices, draw_data->TotalVtxCount);
    }
    if (smoke && ++frames == warmup_frames + measured_frames) {
      glFinish();
      const double elapsed = glfwGetTime() - measure_start;
      const GLenum error = glGetError();
      if (error != GL_NO_ERROR) {
        std::fprintf(stderr, "Smoke test encountered OpenGL error 0x%04x.\n",
                     error);
        return 1;
      }
      if (app.zoom() > wildlife::App::max_zoom) {
        std::fputs("City/state zoom cap exceeded.\n", stderr);
        return 1;
      }
      std::printf("Rendered %d %s frames at %dx%d in %.3f s "
                  "(%.1f FPS, peak %d ImGui vertices/frame).\n",
                  measured_frames,
                  mode == RunMode::DotsSmoke       ? "particle map"
                  : mode == RunMode::ZoomSmoke     ? "globe-to-regional zoom"
                  : mode == RunMode::ForecastSmoke ? "model forecast"
                                                   : "globe",
                  width, height, elapsed, measured_frames / elapsed,
                  peak_vertices);
      std::printf("Observed animals: %zu | Cohorts: %zu | Catalog taxa: %zu | "
                  "Zoom: %.1f / %.1f\n",
                  app.observed_animals(), app.cohort_count(),
                  app.recorded_taxa(), app.zoom(), wildlife::App::max_zoom);
      std::printf("Scenario generation: %d\n", app.scenario_generation());
      if (const char *required = std::getenv("ATLAS_TEST_SCENARIO_GENERATION"))
        if (app.scenario_generation() != std::atoi(required)) {
          std::fputs("Continuous scenario did not reach its expected GPU "
                     "generation.\n",
                     stderr);
          return 1;
        }
      return 0;
    }
  }

  if (smoke) {
    std::fputs("Smoke test closed before all frames were rendered.\n", stderr);
    return 1;
  }
  return 0;
}

int run(GLFWwindow *window, const char *glsl_version, RunMode mode,
        const char *executable, std::string_view test_species) {
  glfwMakeContextCurrent(window);
  std::printf("OpenGL renderer: %s\n", glGetString(GL_RENDERER));
  glfwSwapInterval(mode == RunMode::Interactive ? 1 : 0);

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::StyleColorsDark();
  ImGui::GetStyle().FrameRounding = 3.0f;
  ImGui::GetStyle().FramePadding = {5.0f, 3.0f};
  ImGuiIO &io = ImGui::GetIO();
  io.IniFilename = nullptr;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

  int result = 1;
  if (ImGui_ImplGlfw_InitForOpenGL(window, true)) {
    if (ImGui_ImplOpenGL3_Init(glsl_version)) {
      if (ImGui_ImplOpenGL3_CreateDeviceObjects()) {
        const bool has_terrain = wildlife::initialize_earth_texture(
            executable,
            mode == RunMode::Interactive || std::getenv("ATLAS_STREAM_TILES"));
        if (!has_terrain) {
          std::fputs("Could not initialize the Earth terrain. Check the "
                     "bundled elevation and map textures.\n",
                     stderr);
        }
        if (has_terrain || mode == RunMode::Interactive) {
          result = render_loop(window, mode, executable, test_species);
        }
        wildlife::shutdown_earth_texture();
      }
      ImGui_ImplOpenGL3_Shutdown();
    } else {
      std::fputs("Unable to initialize the Dear ImGui OpenGL backend.\n",
                 stderr);
    }
    ImGui_ImplGlfw_Shutdown();
  } else {
    std::fputs("Unable to initialize the Dear ImGui GLFW backend.\n", stderr);
  }
  ImGui::DestroyContext();
  return result;
}

} // namespace

int main(int argc, char **argv) {
  RunMode mode = RunMode::Interactive;
  std::string_view test_species;
  if (argc > 1) {
    const std::string_view option = argv[1];
    if (argc == 2 && (option == "--help" || option == "-h")) {
      std::puts("Usage: wildlife_atlas [--smoke-test [dots | zoom | forecast] "
                "[species]]\nSet ATLAS_FORECAST to "
                "load a WildCast JSON on startup.");
      return 0;
    }
    if (argc == 2 && option == "--smoke-test") {
      mode = RunMode::GlobeSmoke;
    } else if ((argc == 3 || argc == 4) && option == "--smoke-test") {
      const std::string_view kind{argv[2]};
      if (kind == "dots" || kind == "surface" || kind == "colony")
        mode = RunMode::DotsSmoke;
      else if (kind == "zoom")
        mode = RunMode::ZoomSmoke;
      else if (kind == "forecast")
        mode = RunMode::ForecastSmoke;
      else {
        std::fputs("Unknown smoke-test mode; use dots, zoom or forecast.\n",
                   stderr);
        return 2;
      }
      if (argc == 4)
        test_species = argv[3];
    } else {
      std::fputs("Unknown option; use --help.\n", stderr);
      return 2;
    }
  }

  glfwSetErrorCallback(glfw_error);
  if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK)
    return 1;
  if (!glfwInit()) {
    return 1;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_DEPTH_BITS, 24);
  glfwWindowHint(GLFW_SAMPLES, 4);
#ifdef __APPLE__
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#endif
  const char *glsl_version = "#version 150";
  if (mode != RunMode::Interactive) {
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
  }

  int result = 1;
  if (GLFWwindow *window = create_window()) {
    result = run(window, glsl_version, mode, argv[0], test_species);
    glfwDestroyWindow(window);
  }
  glfwTerminate();
  curl_global_cleanup();
  return result;
}
