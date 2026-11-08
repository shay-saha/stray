#include "simulation.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <numbers>
#include <set>

namespace wildlife {
namespace {
constexpr double rad = std::numbers::pi / 180., earth_km = 6371.;
Vec3 unit(Vec3 p) {
  const double length = std::hypot(double(p.x), double(p.y), double(p.z));
  return {float(p.x / length), float(p.y / length), float(p.z / length)};
}
} // namespace
Vec3 earth_position(float lat, float lon) {
  const double a = lat * rad, b = lon * rad;
  return {float(std::cos(a) * std::cos(b)), float(std::sin(a)),
          float(std::cos(a) * std::sin(b))};
}
float latitude_degrees(Vec3 p) {
  return float(std::atan2(double(p.y), std::hypot(double(p.x), double(p.z))) /
               rad);
}
float longitude_degrees(Vec3 p) {
  return float(std::atan2(double(p.z), double(p.x)) / rad);
}
std::string_view behavior_name(Behavior) { return "Tracking cohort"; }
std::string_view activity_name(Activity) { return "Movement"; }

bool Simulation::replace_tracks(std::vector<TrackedAnimal> tracks,
                                bool predicted, std::vector<Species> catalog) {
  std::set<std::string> identities;
  for (const auto &track : tracks) {
    if (track.id.empty() || track.species.empty() || track.hours.empty() ||
        track.positions.size() != track.hours.size() ||
        !identities.insert(track.id).second)
      return false;
    for (std::size_t i = 0; i < track.hours.size(); ++i) {
      const auto p = track.positions[i];
      if (!std::isfinite(track.hours[i]) ||
          (i && track.hours[i] <= track.hours[i - 1]) || !std::isfinite(p.x) ||
          !std::isfinite(p.y) || !std::isfinite(p.z) ||
          std::hypot(p.x, p.y, p.z) < .5)
        return false;
    }
  }
  std::vector<Species> species = std::move(catalog);
  std::vector<Colony> colonies;
  std::vector<std::vector<std::size_t>> members;
  std::map<std::string, std::size_t> species_ids;
  for (std::size_t i = 0; i < species.size(); ++i) {
    if (species[i].scientific_name.empty() ||
        !species_ids.emplace(species[i].scientific_name, i).second)
      return false;
  }
  std::map<std::pair<std::string, std::size_t>, std::size_t> cohort_ids;
  for (std::size_t i = 0; i < tracks.size(); ++i) {
    auto &track = tracks[i];
    for (auto &p : track.positions)
      p = unit(p);
    const auto [taxon, added] =
        species_ids.try_emplace(track.species, species.size());
    if (added)
      species.push_back({track.name.empty() ? track.species : track.name,
                         Behavior::Herd,
                         track.group,
                         track.species,
                         {}});
    const auto [cohort, new_cohort] = cohort_ids.try_emplace(
        std::pair{track.cohort_id, taxon->second}, colonies.size());
    if (new_cohort) {
      colonies.push_back(
          {taxon->second, track.positions.front(), track.cohort_label, 0, 0});
      members.emplace_back();
    }
    members[cohort->second].push_back(i);
    ++colonies[cohort->second].population;
  }
  tracks_ = std::move(tracks);
  species_ = std::move(species);
  colonies_ = std::move(colonies);
  members_ = std::move(members);
  predicted_ = predicted;
  seek(0);
  homes_.clear();
  for (const auto &colony : colonies_)
    homes_.push_back(colony.position);
  return true;
}

AnimalSample Simulation::sample_animal(AnimalId animal, double hours) const {
  const auto &track =
      tracks_.at(members_.at(animal.colony).at(animal.individual));
  const auto next =
      std::upper_bound(track.hours.begin(), track.hours.end(), hours);
  Vec3 p;
  if (next == track.hours.begin())
    p = track.positions.front();
  else if (next == track.hours.end())
    p = track.positions.back();
  else {
    const std::size_t i = std::size_t(next - track.hours.begin());
    const double t =
        (hours - track.hours[i - 1]) / (track.hours[i] - track.hours[i - 1]);
    const auto a = track.positions[i - 1], b = track.positions[i];
    p = {float(std::lerp(double(a.x), double(b.x), t)),
         float(std::lerp(double(a.y), double(b.y), t)),
         float(std::lerp(double(a.z), double(b.z), t))};
    if (std::hypot(p.x, p.y, p.z) < 1e-6)
      p = t < .5 ? a : b;
    else
      p = unit(p);
  }
  return {p, .88f, Activity::Traveling};
}
std::size_t Simulation::sample_colony(std::size_t colony, double hours,
                                      std::span<AnimalSample> output) const {
  const auto count = std::min(output.size(), members_.at(colony).size());
  for (std::size_t i = 0; i < count; ++i)
    output[i] = sample_animal({colony, i}, hours);
  return count;
}
void Simulation::sample_trail(AnimalId animal, double hours,
                              double history_hours,
                              std::vector<AnimalTrailSegment> &segments) const {
  segments.clear();
  if (history_hours <= 0)
    return;
  const auto &track =
      tracks_.at(members_.at(animal.colony).at(animal.individual));
  const double begin = hours - history_hours;
  const auto next =
      std::upper_bound(track.hours.begin(), track.hours.end(), begin);
  for (std::size_t i = std::max<std::size_t>(1, next - track.hours.begin());
       i < track.hours.size(); ++i) {
    const double a = std::max(begin, track.hours[i - 1]),
                 b = std::min(hours, track.hours[i]);
    if (a >= hours)
      break;
    if (b <= a)
      continue;
    // Missing observations do not establish a traveled route across a long data
    // gap.
    if (track.hours[i] <= track.forecast_start_hour &&
        track.hours[i] - track.hours[i - 1] > 6)
      continue;
    segments.push_back({sample_animal(animal, a).position,
                        sample_animal(animal, b).position, a, b});
  }
}
void Simulation::sample_trail_points(
    AnimalId animal, double hours, double history_hours, std::size_t budget,
    std::vector<AnimalTrailPoint> &points) const {
  points.clear();
  if (!budget || !std::isfinite(hours) || !std::isfinite(history_hours) ||
      history_hours <= 0)
    return;
  const auto &track =
      tracks_.at(members_.at(animal.colony).at(animal.individual));
  const double span = std::min(history_hours, hours - track.hours.front());
  if (span <= 0)
    return;
  Vec3 previous = sample_animal(animal, hours).position;
  double previous_hour = std::min(hours, track.hours.back());
  const auto emit = [&](double at) {
    if (at > track.hours.back())
      return;
    const auto next =
        std::upper_bound(track.hours.begin(), track.hours.end(), at);
    if (next != track.hours.begin() && next != track.hours.end() &&
        *next <= track.forecast_start_hour && *next - *(next - 1) > 6)
      return;
    const auto p = sample_animal(animal, at).position;
    if (std::hypot(double(p.x - previous.x), double(p.y - previous.y),
                   double(p.z - previous.z)) < 1e-7)
      return;
    bool connected = true;
    if (at < track.forecast_start_hour) {
      // Inspect all crossed observation intervals, including gaps that fall
      // between two valid samples in a long, downsampled history.
      for (auto end = next;
           end != track.hours.end() && *end <= track.forecast_start_hour &&
           *(end - 1) < previous_hour; ++end) {
        if (*end - *(end - 1) > 6) {
          connected = false;
          break;
        }
      }
    }
    points.push_back({p, at, connected});
    previous = p;
    previous_hour = at;
  };
  const double begin_hour = hours - span;
  const auto begin = std::lower_bound(track.hours.begin(), track.hours.end(), begin_hour);
  const auto end = std::lower_bound(track.hours.begin(), track.hours.end(), hours);
  const bool clipped = begin != track.hours.end() && *begin > begin_hour;
  if (std::size_t(end - begin) + std::size_t(clipped) <= budget) {
    // A week/month usually contains only a handful of model horizons. Retain
    // every real knot rather than spending the budget on redundant resamples.
    for (auto at = end; at != begin;)
      emit(*--at);
    if (clipped)
      emit(begin_hour);
  } else {
    for (std::size_t sample = 1; sample <= budget; ++sample) {
      const double age = double(sample) / budget;
      emit(hours - span * age * age * age);
    }
  }
}
void Simulation::seek(double hours) {
  if (!std::isfinite(hours) || hours < 0)
    return;
  elapsed_hours_ = hours;
  for (std::size_t c = 0; c < colonies_.size(); ++c) {
    double x = 0, y = 0, z = 0;
    for (std::size_t i = 0; i < members_[c].size(); ++i) {
      const auto p = sample_animal({c, i}, hours).position;
      x += p.x;
      y += p.y;
      z += p.z;
    }
    auto &colony = colonies_[c];
    const double n = std::hypot(x, y, z);
    colony.position = n > 1e-9 ? Vec3{float(x / n), float(y / n), float(z / n)}
                               : sample_animal({c, 0}, hours).position;
    double extent = 0;
    for (std::size_t i = 0; i < members_[c].size(); ++i) {
      const auto p = sample_animal({c, i}, hours).position,
                 center = colony.position;
      extent = std::max(extent, 2 * earth_km *
                                    std::asin(std::min(
                                        1., std::hypot(double(p.x - center.x),
                                                       double(p.y - center.y),
                                                       double(p.z - center.z)) *
                                                .5)));
    }
    colony.draw_radius_km = float(extent + 1);
  }
}
std::string_view Simulation::animal_name(AnimalId id) const {
  return tracks_.at(members_.at(id.colony).at(id.individual)).id;
}
std::optional<AnimalId> Simulation::find_animal(std::string_view name) const {
  for (std::size_t c = 0; c < members_.size(); ++c)
    for (std::size_t i = 0; i < members_[c].size(); ++i)
      if (animal_name({c, i}) == name)
        return AnimalId{c, i};
  return {};
}
std::optional<AnimalId> Simulation::set_forecast(std::string_view species,
                                                 std::vector<double> hours,
                                                 std::vector<Vec3> positions,
                                                 double start) {
  if (species.empty() || !std::isfinite(start) || start < 0 || hours.empty() ||
      hours.front() != 0)
    return {};
  for (auto &hour : hours)
    hour += start;
  TrackedAnimal track{"tracked-animal",     std::string(species),
                      std::string(species), std::string(species),
                      "Observed animal",    AnimalGroup::OtherMammal,
                      std::move(hours),     std::move(positions)};
  if (!replace_tracks({std::move(track)}, true))
    return {};
  seek(start);
  return AnimalId{0, 0};
}
void Simulation::clear_forecast() { replace_tracks({}, false); }
} // namespace wildlife
