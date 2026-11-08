#include "simulation.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

void check(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
int main() {
  using namespace wildlife;
  Simulation s;
  check(s.species().empty() && s.colonies().empty() && s.animal_count() == 0,
        "Empty must never create synthetic animals");
  const auto a = earth_position(0, 179), b = earth_position(0, -179);
  auto id = s.set_forecast("Ciconia ciconia", {0, 24}, {a, b}, 100);
  check(id.has_value(), "Observed taxon should load");
  check(s.animal_count() == 1 && s.colonies()[0].population == 1,
        "Uncertainty must not become extra animals");
  check(
      std::abs(std::abs(longitude_degrees(s.sample_animal(*id, 112).position)) -
               180) < .001,
      "Antimeridian takes short arc");
  check(std::abs(longitude_degrees(s.sample_animal(*id, 90).position) - 179) <
            .001,
        "Before-origin time clamps");
  check(std::abs(longitude_degrees(s.sample_animal(*id, 9999).position) + 179) <
            .001,
        "Routes must not loop annually");
  check(!s.set_forecast("Ciconia ciconia", {0, 0}, {a, b}, 100),
        "Duplicate horizons rejected");
  check(!s.set_forecast("Ciconia ciconia",
                        {0, std::numeric_limits<double>::quiet_NaN()}, {a, b},
                        100),
        "Invalid horizons rejected");
  std::vector<TrackedAnimal> tracks;
  for (int i = 0; i < 1200; ++i)
    tracks.push_back({"test-tag-" + std::to_string(i),
                      "Observed taxon",
                      "Observed taxon",
                      "study|taxon",
                      "Study cohort",
                      AnimalGroup::Bird,
                      {0, 24},
                      {a, b}});
  std::vector<Species> catalog{{"Observed taxon",
                                Behavior::Flock,
                                AnimalGroup::Bird,
                                "Observed taxon",
                                {}},
                               {"Inactive taxon",
                                Behavior::Flock,
                                AnimalGroup::Bird,
                                "Inactive taxon",
                                {}}};
  check(s.replace_tracks(tracks, true, catalog),
        "Observed cohort should replace data atomically");
  check(s.species().size() == 2 && s.animal_count() == 1200 &&
            s.colonies().size() == 1,
        "Inactive catalog labels must not create animals");
  std::vector<AnimalSample> samples(1200);
  check(s.sample_colony(0, 12, samples) == 1200,
        "All tracked animals render, including beyond 512");
  check(s.animal_name({0, 1199}) == "test-tag-1199",
        "Every marker retains source identity");
  s.seek(12);
  check(std::abs(std::abs(longitude_degrees(s.colonies()[0].position)) - 180) <
            .001,
        "Cohort culling center follows member positions");
  tracks[1].id = tracks[0].id;
  check(!s.replace_tracks(tracks, true),
        "Duplicate identities must not duplicate animals");
  check(s.animal_count() == 1200,
        "A failed import must retain the old valid data");
  check(s.set_forecast("Previously unseen observed taxon", {0, 24}, {a, b}, 0)
            .has_value(),
        "Particle rendering must not require a procedural animal profile");
  s.clear_forecast();
  check(s.animal_count() == 0 && s.species().empty() && !s.is_forecast(0),
        "Clearing must not generate replacement animals");
  TrackedAnimal trail{
      "trail-tag", "Observed taxon",  "Observed taxon",    "study",
      "study",     AnimalGroup::Bird, {-48, -1, 0, 1, 12}, {a, b, a, b, a}};
  trail.forecast_start_hour = 0;
  check(s.replace_tracks({trail}, true), "Timed trail fixture should load");
  std::vector<AnimalTrailSegment> segments;
  s.sample_trail({0, 0}, 6, 54, segments);
  check(
      segments.size() == 3,
      "Do not join across missing observed history or shortcut recorded turns");
  check(segments.front().start_hour == -1 && segments.back().end_hour == 6,
        "Trail clips to exact time window");
  s.sample_trail({0, 0}, 0, 0, segments);
  check(segments.empty(), "Disabled trails produce no geometry");
  std::cout << "Observed identity, full-cohort counts, route interpolation and "
               "no-synthetic checks passed\n";
}
