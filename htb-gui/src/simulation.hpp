#pragma once
#include "animal_profile.hpp"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace wildlife {
struct Vec2 {
  float x, y;
};
struct Vec3 {
  float x, y, z;
};
enum class Behavior { Herd, Flock, School, Pack, Solitary };
enum class Activity { Traveling, Foraging, Resting };
enum class AnimalGroup {
  Bird,
  Grazer,
  LandCarnivore,
  Primate,
  OtherMammal,
  AquaticMammal,
  FishAndKrill,
  Reptile,
  Count
};
struct Species {
  std::string name;
  Behavior behavior = Behavior::Herd;
  AnimalGroup group = AnimalGroup::OtherMammal;
  std::string scientific_name;
  AnimalProfile appearance{};
};
struct Colony {
  std::size_t species;
  Vec3 position;
  std::string region;
  int population;
  float draw_radius_km;
};
struct AnimalId {
  std::size_t colony, individual;
  bool operator==(const AnimalId &) const = default;
};
struct AnimalSample {
  Vec3 position;
  float brightness;
  Activity activity;
};
struct TrackedAnimal {
  std::string id, species, name, cohort_id, cohort_label;
  AnimalGroup group = AnimalGroup::OtherMammal;
  std::vector<double> hours;
  std::vector<Vec3> positions;
  double forecast_start_hour = std::numeric_limits<double>::infinity();
};
struct AnimalTrailSegment {
  Vec3 start, end;
  double start_hour, end_hour;
};
struct AnimalTrailPoint {
  Vec3 position;
  double hour;
};
Vec3 earth_position(float latitude_deg, float longitude_deg);
float latitude_degrees(Vec3 position);
float longitude_degrees(Vec3 position);
std::string_view behavior_name(Behavior behavior);
std::string_view activity_name(Activity activity);

// Contains observed identities and model routes only. Empty means no animals,
// rather than a generated catalog or a fallback population.
class Simulation {
public:
  static constexpr std::size_t max_visible_animals =
      512; // legacy API sizing; not a rendering limit
  static constexpr double year_hours = 365.0 * 24.0;
  Simulation() = default;
  const std::vector<Species> &species() const { return species_; }
  const std::vector<Colony> &colonies() const { return colonies_; }
  Vec3 colony_home(std::size_t colony) const { return homes_.at(colony); }
  double elapsed_hours() const { return elapsed_hours_; }
  std::size_t animal_count() const { return tracks_.size(); }
  std::string_view animal_name(AnimalId id) const;
  std::optional<AnimalId> find_animal(std::string_view name) const;
  std::size_t sample_colony(std::size_t colony, double hours,
                            std::span<AnimalSample> output) const;
  AnimalSample sample_animal(AnimalId animal, double hours) const;
  void sample_trail(AnimalId animal, double hours, double history_hours,
                    std::vector<AnimalTrailSegment> &segments) const;
  void sample_trail_points(AnimalId animal, double hours, double history_hours,
                           std::size_t budget,
                           std::vector<AnimalTrailPoint> &points) const;
  void seek(double hours);
  bool replace_tracks(std::vector<TrackedAnimal> tracks, bool predicted,
                      std::vector<Species> catalog = {});
  std::optional<AnimalId> set_forecast(std::string_view scientific_name,
                                       std::vector<double> hours,
                                       std::vector<Vec3> positions,
                                       double start);
  void clear_forecast();
  bool is_forecast(std::size_t colony) const {
    return predicted_ && colony < colonies_.size();
  }

private:
  double elapsed_hours_ = 0;
  bool predicted_ = false;
  std::vector<Species> species_;
  std::vector<Colony> colonies_;
  std::vector<TrackedAnimal> tracks_;
  std::vector<std::vector<std::size_t>> members_;
  std::vector<Vec3> homes_;
};
} // namespace wildlife
