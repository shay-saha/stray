#pragma once
#include "globe.hpp"
#include <atomic>
#include <functional>
#include <future>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>

namespace wildlife {
struct ForecastClock {
  int year;
  double hours, begin, end;
};
class ForecastController {
public:
  ForecastController();
  ~ForecastController();
  void poll();
  std::optional<AnimalId> apply(Simulation &, std::vector<std::uint8_t> &);
  std::optional<AnimalId> controls(Simulation &, std::vector<std::uint8_t> &);
  void select(const Simulation &, std::optional<AnimalId>);
  void draw(ImDrawList &, const GlobeView &, const Simulation &) const;
  bool ready() const { return !result_.is_null(); }
  bool predicted() const {
    return ready() && result_.value("kind", "") == "forecast";
  }
  bool pending() const { return request_.valid() || !job_id_.empty(); }
  bool load(const std::string &);
  std::optional<ForecastClock> clock(double hours) const;
  void seek_relative(Simulation &, double hours) const;
  void tick(Simulation &);
  bool scenario() const { return ready() && result_.contains("scenario"); }
  int scenario_generation() const {
    return scenario() ? result_["scenario"].value("generation", 0) : -1;
  }

private:
  enum class Request { Connect, Start, Job };
  void accept(nlohmann::json);
  void load_archive(nlohmann::json, const std::string &path);
  void start_forecast();
  void start_scenario(bool extend);
  std::string url_, status_ = "Connecting to local Stray model...", job_id_,
                    selected_id_;
  std::future<nlohmann::json> request_;
  std::shared_ptr<std::atomic_bool> stop_ =
      std::make_shared<std::atomic_bool>(false);
  Request request_kind_ = Request::Connect;
  bool apply_ = false, manual_loaded_ = false, show_paths_ = false,
       show_history_ = false;
  nlohmann::json result_, health_;
  nlohmann::json next_scenario_;
  std::vector<TrackedAnimal> imported_tracks_;
  bool continuing_ = false, scenario_failed_ = false, discard_request_ = false;
  double resume_hours_ = 0;
  int mode_ = 0;
  char observations_[2048]{}, issue_time_[80]{},
      json_path_[2048] = "runs/latest-cohort-forecast.json";
  int samples_ = 16, max_age_hours_ = 24;
  std::size_t unknown_species_ = 0, limited_history_ = 0;
  double origin_seconds_ = 0, end_days_ = 0, next_poll_ = 0;
};
} // namespace wildlife
