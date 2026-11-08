#pragma once

#include "forecast.hpp"
#include "habitat_view.hpp"
#include "population_view.hpp"

#include <cstdint>
#include <optional>
#include <vector>

namespace wildlife {

class App {
public:
  explicit App(const std::filesystem::path &executable = {});
  static constexpr float min_zoom = 0.75f;
  static constexpr float max_zoom = 160.0f;
  void draw();
  void focus_population();
  bool focus_dots(std::string_view name, float zoom = max_zoom);
  void test_zoom(float zoom);
  bool load_forecast(const std::string &path);
  bool model_ready() const {
    return forecast_.predicted() && simulation_.animal_count() > 0;
  }
  std::size_t observed_animals() const { return simulation_.animal_count(); }
  std::size_t recorded_taxa() const { return simulation_.species().size(); }
  std::size_t cohort_count() const { return simulation_.colonies().size(); }
  float zoom() const { return globe_zoom_; }
  int scenario_generation() const { return forecast_.scenario_generation(); }
  void test_forecast_day(double day, bool play) {
    forecast_.seek_relative(simulation_,
                            day * 24 - simulation_.elapsed_hours());
    paused_ = !play;
  }
  void focus_location(double latitude, double longitude, float zoom) {
    reset_camera();
    selected_.reset();
    auto_rotate_ = false;
    view_latitude_ = latitude;
    view_longitude_ = longitude;
    test_zoom(zoom);
  }

private:
  void draw_toolbar();
  void draw_sidebar();
  void draw_map();
  void draw_globe(ImDrawList &draw, ImVec2 min, ImVec2 max,
                  MapLabelRect calendar, bool hovered);
  void reset_camera();
  void focus(Vec3 position, float zoom);
  void locate_species(std::size_t species);
  void focus_group(std::size_t colony);

  Simulation simulation_;
  ForecastController forecast_;
  std::vector<std::uint8_t> enabled_;
  ImGuiTextFilter species_filter_;
  std::vector<std::size_t> listed_species_;
  ImFont *habitat_font_ = nullptr;
  ImFontAtlasRectId grain_sprite_ = ImFontAtlasRectId_Invalid;
  std::optional<AnimalId> selected_;
  double rotation_resume_time_ = 0.0;
  double view_latitude_ = 24.0;
  double view_longitude_ = -24.0;
  float globe_zoom_ = 1.0f;
  float target_zoom_ = 1.0f;
  float terrain_angle_ = 28.0f;
  int speed_index_ = 3;
  int trail_index_ = 1;
  bool paused_ = true;
  bool auto_rotate_ = false;
  bool night_view_ = false;
  bool terrain_tilt_ = true;
  bool both_hemispheres_ = false;
  bool following_ = false;
  bool calendar_dragging_ = false;
  bool map_dragged_ = false;
  bool zoomed_on_click_ = false;
  bool drag_opposite_ = false;
};

} // namespace wildlife
