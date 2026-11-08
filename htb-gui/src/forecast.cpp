#include "forecast.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <curl/curl.h>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>

namespace wildlife {
namespace {
using json = nlohmann::json;
std::size_t receive(char *data, std::size_t size, std::size_t count,
                    void *target) {
  auto &body = *static_cast<std::string *>(target);
  const auto bytes = size * count;
  if (body.size() + bytes > 128 * 1024 * 1024)
    return 0;
  body.append(data, bytes);
  return bytes;
}
json http(const std::string &url, const json *payload = nullptr,
          std::atomic_bool *stop = nullptr) {
  CURL *curl = curl_easy_init();
  if (!curl)
    throw std::runtime_error("Could not initialize inference connection");
  std::string body, request = payload ? payload->dump() : "";
  auto *headers = curl_slist_append(nullptr, "Content-Type: application/json");
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt(curl, CURLOPT_NOPROXY, "127.0.0.1,localhost");
  curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 2L);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 180L);
  curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, receive);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &body);
  if (stop) {
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(curl, CURLOPT_XFERINFODATA, stop);
    curl_easy_setopt(
        curl, CURLOPT_XFERINFOFUNCTION,
        +[](void *state, curl_off_t, curl_off_t, curl_off_t,
            curl_off_t) -> int {
          return static_cast<std::atomic_bool *>(state)->load() ? 1 : 0;
        });
  }
  if (payload)
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, request.c_str());
  const auto code = curl_easy_perform(curl);
  long status = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
  const std::string error = curl_easy_strerror(code);
  curl_slist_free_all(headers);
  curl_easy_cleanup(curl);
  if (code != CURLE_OK)
    throw std::runtime_error(error +
                             ". Start with ./run-local to enable inference.");
  const auto value = json::parse(body);
  if (status != 200 && status != 202)
    throw std::runtime_error(value.value("error", "Inference service error"));
  return value;
}
template <std::size_t N>
void copy(char (&buffer)[N], const std::string &value) {
  std::snprintf(buffer, N, "%s", value.c_str());
}
Vec3 coordinate(const json &p) {
  return earth_position(p.at(0).get<float>(), p.at(1).get<float>());
}
void point(const json &p) {
  if (!p.is_array() || p.size() != 2 || !p[0].is_number() || !p[1].is_number())
    throw std::runtime_error("Invalid forecast coordinate");
  const double lat = p[0], lon = p[1];
  if (!std::isfinite(lat) || !std::isfinite(lon) || std::abs(lat) > 90 ||
      std::abs(lon) > 180)
    throw std::runtime_error("Forecast coordinate outside Earth");
}
void line(ImDrawList &draw, const std::function<bool(Vec3, ImVec2 &)> &project,
          Vec3 a, Vec3 b, ImU32 color, float width) {
  // Subdivide great-circle chords; don't connect through the back of the
  // planet.
  const double dot = std::clamp(
      double(a.x) * b.x + double(a.y) * b.y + double(a.z) * b.z, -1., 1.);
  const int steps = std::max(1, int(std::acos(dot) * 180 / 3.141592653589793));
  ImVec2 previous;
  bool visible = project(a, previous);
  for (int i = 1; i <= steps; ++i) {
    const float t = float(i) / steps;
    Vec3 v{std::lerp(a.x, b.x, t), std::lerp(a.y, b.y, t),
           std::lerp(a.z, b.z, t)};
    const float n = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    if (n < .001f) {
      visible = false;
      continue;
    }
    v = {v.x / n, v.y / n, v.z / n};
    ImVec2 current;
    const bool next = project(v, current);
    if (visible && next)
      draw.AddLine(previous, current, color, width);
    previous = current;
    visible = next;
  }
}
} // namespace
namespace {
double timestamp(const std::string &stamp) {
  int y = 0, m = 0, d = 0, hh = 0, mm = 0;
  double ss = 0;
  if (std::sscanf(stamp.c_str(), "%d-%d-%dT%d:%d:%lf", &y, &m, &d, &hh, &mm,
                  &ss) != 6 ||
      (stamp.back() != 'Z' && !stamp.ends_with("+00:00")))
    throw std::runtime_error("Expected an ISO UTC timestamp");
  const auto date = std::chrono::year{y} / std::chrono::month{unsigned(m)} /
                    std::chrono::day{unsigned(d)};
  if (!date.ok() || hh < 0 || hh > 23 || mm < 0 || mm > 59 ||
      !std::isfinite(ss) || ss < 0 || ss >= 60)
    throw std::runtime_error("Invalid UTC timestamp");
  return std::chrono::duration<double>(
             std::chrono::sys_days{date}.time_since_epoch())
             .count() +
         hh * 3600 + mm * 60 + ss;
}
AnimalGroup color_family(const std::string &taxon) {
  const auto genus = taxon.substr(0, taxon.find(' '));
  for (const auto g : {"Antilocapra", "Cervus", "Equus", "Odocoileus", "Ovis",
                       "Rangifer", "Syncerus", "Sus", "Loxodonta"})
    if (genus == g)
      return AnimalGroup::Grazer;
  for (const auto g : {"Canis", "Panthera", "Puma", "Lynx", "Vulpes", "Ursus",
                       "Caracal", "Gulo", "Nyctereutes", "Pekania"})
    if (genus == g)
      return AnimalGroup::LandCarnivore;
  for (const auto g : {"Halichoerus", "Physeter", "Balaenoptera", "Zalophus"})
    if (genus == g)
      return AnimalGroup::AquaticMammal;
  for (const auto g : {"Chelonia", "Eretmochelys", "Testudinidae"})
    if (genus == g)
      return AnimalGroup::Reptile;
  if (genus == "Manta" || genus == "Mobula")
    return AnimalGroup::FishAndKrill;
  if (genus == "Papio")
    return AnimalGroup::Primate;
  for (const auto g :
       {"Brachylagus", "Tapirus", "Wallabia", "Eidolon", "Hypsignathus",
        "Lavia", "Noctilio", "Phyllostomus", "Pteropus", "Tadarida",
        "Phascolarctos"})
    if (genus == g)
      return AnimalGroup::OtherMammal;
  return AnimalGroup::Bird;
}
} // namespace
ForecastController::ForecastController() {
  const char *env = std::getenv("ATLAS_INFERENCE_URL");
  url_ = env ? env : "http://127.0.0.1:8765";
  request_ = std::async(std::launch::async, [url = url_, stop = stop_] {
    auto h = http(url + "/health", nullptr, stop.get());
    if (h.value("service", "") != "wildlife-atlas-wildcast-v1" ||
        h.value("api_version", 0) < 2)
      throw std::runtime_error(
          "Restart ./run-local to use the observed-cohort service");
    if (h.value("scenario_api_version", 0) < 1)
      throw std::runtime_error(
          "Restart ./run-local to enable continuous model scenarios");
    return json{{"health", h}};
  });
}
ForecastController::~ForecastController() { stop_->store(true); }
void ForecastController::start_forecast() {
  next_scenario_ = nullptr;
  scenario_failed_ = false;
  continuing_ = false;
  const json input{{"observations", observations_},
                   {"issue_time", issue_time_},
                   {"samples", samples_},
                   {"max_age_hours", max_age_hours_}};
  status_ = "Preparing forecasts for all eligible recorded animals...";
  request_kind_ = Request::Start;
  request_ = std::async(std::launch::async, [url = url_, input, stop = stop_] {
    return http(url + "/forecast-cohorts", &input, stop.get());
  });
}
void ForecastController::start_scenario(bool extend) {
  json input;
  if (extend) {
    input = {{"scenario_id", result_["scenario"]["id"]},
             {"generation", scenario_generation()}};
    status_ = "Forecasting the next scenario window on GPU...";
  } else {
    input = {{"observations", observations_},
             {"issue_time", issue_time_},
             {"samples", samples_}};
    next_scenario_ = nullptr;
    scenario_failed_ = false;
    continuing_ = false;
    status_ = "Initializing a live model scenario from recorded identities...";
  }
  request_kind_ = Request::Start;
  request_ =
      std::async(std::launch::async, [url = url_, input, extend, stop = stop_] {
        return http(url + (extend ? "/scenario/advance" : "/scenario"), &input,
                    stop.get());
      });
}
void ForecastController::tick(Simulation &simulation) {
  if (!scenario())
    return;
  const double step = result_["scenario"].value("step_days", 30.) * 24;
  if (!next_scenario_.is_null() &&
      simulation.elapsed_hours() >= step - .000001) {
    resume_hours_ =
        std::max(0., simulation.elapsed_hours() + origin_seconds_ / 3600 -
                         timestamp(next_scenario_["issue_time"]) / 3600);
    accept(std::move(next_scenario_));
    next_scenario_ = nullptr;
    continuing_ = true;
    return;
  }
  if (!scenario_failed_ && !apply_ && !pending() && next_scenario_.is_null())
    start_scenario(true);
}
void ForecastController::poll() {
  if (!request_.valid()) {
    if (!job_id_.empty() && ImGui::GetTime() >= next_poll_) {
      request_kind_ = Request::Job;
      next_poll_ = ImGui::GetTime() + .5;
      request_ = std::async(
          std::launch::async, [url = url_, id = job_id_, stop = stop_] {
            return http(url + "/jobs/" + id, nullptr, stop.get());
          });
    }
    return;
  }
  if (request_.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
    return;
  try {
    auto value = request_.get();
    if (discard_request_) {
      discard_request_ = false;
      return;
    }
    if (request_kind_ == Request::Connect) {
      health_ = value.at("health");
      if (!manual_loaded_) {
        start_scenario(false);
      }
    } else if (request_kind_ == Request::Start) {
      job_id_ = value.at("job_id");
      next_poll_ = 0;
    } else {
      const auto state = value.value("state", "");
      if (state == "complete") {
        job_id_.clear();
        auto result = std::move(value.at("result"));
        if (scenario() && result.contains("scenario") &&
            result["scenario"]["id"] == result_["scenario"]["id"] &&
            result["scenario"].value("generation", 0) > scenario_generation()) {
          next_scenario_ = std::move(result);
          status_ = "Next GPU forecast ready; playback will continue across "
                    "the month boundary";
        } else
          accept(std::move(result));
      } else if (state == "failed") {
        job_id_.clear();
        throw std::runtime_error(value.value("error", "Forecast failed"));
      } else
        status_ = value.value("phase", "Predicting") + " | " +
                  std::to_string(value.value("completed", 0)) + " / " +
                  std::to_string(value.value("total", 0));
    }
  } catch (const std::exception &error) {
    if (discard_request_) {
      discard_request_ = false;
      return;
    }
    if (manual_loaded_ && request_kind_ == Request::Connect)
      status_ += " | Local inference service offline";
    else
      status_ = error.what();
    job_id_.clear();
    scenario_failed_ = true;
  }
}
void ForecastController::accept(json value) {
  if (!value.contains("schema") && value.contains("sampled_latlon_paths")) {
    json track = value;
    track["id"] = value.contains("selected_animal") &&
                          value["selected_animal"].is_string()
                      ? value["selected_animal"]
                      : json("imported-animal");
    track["cohort_id"] = value.at("species");
    track["cohort_label"] = "Imported observed animal";
    track["representative_sample"] =
        value.value("atlas_representative_sample", std::size_t(0));
    value = {
        {"schema", "wildlife-atlas-cohorts-v2"},
        {"kind", "forecast"},
        {"issue_time", value["origin"]["timestamp"]},
        {"tracks", json::array({track})},
        {"forecast_end_days", value["horizons_days"].back()},
        {"runtime", value.value("runtime", json::object())},
        {"retrospective_replay", value.value("retrospective_replay", true)}};
  }
  if (value.value("schema", "") != "wildlife-atlas-cohorts-v2" ||
      !value.at("tracks").is_array())
    throw std::runtime_error("Expected an observed-cohort forecast file");
  const double issue = timestamp(value.at("issue_time"));
  const bool forecast = value.value("kind", "") == "forecast";
  std::set<std::string> ids;
  for (const auto &track : value["tracks"]) {
    const auto id = track.at("id").get<std::string>();
    if (id.empty() || !ids.insert(id).second ||
        track.at("species").get<std::string>().empty())
      throw std::runtime_error("Animal identities and scientific species must "
                               "be present and unique");
    point(json::array({track["origin"]["lat"], track["origin"]["lon"]}));
    const double origin = timestamp(track["origin"].at("timestamp"));
    if (origin > issue + .001)
      throw std::runtime_error(
          "A forecast origin cannot follow its issue time");
    double previous = -1e30;
    const auto &history = track.value(
        value.contains("scenario") ? "scenario_history" : "observed_history",
        json::array());
    for (const auto &fix : history) {
      point(json::array({fix.at("lat"), fix.at("lon")}));
      const double at = timestamp(fix.at("timestamp"));
      if (at <= previous || at > issue + .001)
        throw std::runtime_error(
            "Observed history must be ordered and exclude future fixes");
      previous = at;
    }
    if (forecast) {
      const auto &h = track.at("horizons_days"),
                 &paths = track.at("sampled_latlon_paths");
      if (h.empty() || h.size() > 64 || paths.empty() || paths.size() > 128)
        throw std::runtime_error("Invalid route dimensions");
      double last = 0;
      for (const auto &day : h) {
        const double d = day;
        if (!std::isfinite(d) || d <= last)
          throw std::runtime_error("Horizon order is invalid");
        last = d;
      }
      for (const auto &path : paths) {
        if (path.size() != h.size())
          throw std::runtime_error("Route/horizon mismatch");
        for (const auto &p : path)
          point(p);
      }
      if (track.value("representative_sample", std::size_t(0)) >= paths.size())
        throw std::runtime_error("Invalid followed route");
    }
  }
  const double end = value.value("forecast_end_days", 0.0);
  if (!std::isfinite(end) || end < 0 || end > 366)
    throw std::runtime_error("Invalid forecast time range");
  result_ = std::move(value);
  imported_tracks_.clear();
  mode_ = scenario() ? 0 : 1;
  unknown_species_ = 0;
  limited_history_ = 0;
  for (const auto &track : result_["tracks"]) {
    unknown_species_ += track.value("unknown_species_fallback", false);
    limited_history_ +=
        forecast && !track.value("history_support_matches_training", true);
  }
  origin_seconds_ = issue;
  end_days_ = end;
  apply_ = true;
  status_ = std::to_string(result_["tracks"].size()) +
            (forecast ? " model-predicted animals" : " observed animals");
  if (result_.contains("dataset"))
    copy(observations_, result_["dataset"].value("observations", ""));
  copy(issue_time_, result_.at("issue_time"));
}
bool ForecastController::load(const std::string &path) {
  try {
    std::ifstream file(path);
    if (!file)
      throw std::runtime_error("Cannot read forecast JSON");
    auto value = json::parse(file);
    if (value.value("schema", "") == "wildlife-atlas-trajectory-archive-v1")
      load_archive(std::move(value), path);
    else
      accept(std::move(value));
    manual_loaded_ = true;
    job_id_.clear();
    next_scenario_ = nullptr;
    continuing_ = false;
    discard_request_ = request_.valid() && request_kind_ != Request::Connect;
    copy(json_path_, path);
    return true;
  } catch (const std::exception &error) {
    status_ = error.what();
    return false;
  }
}
void ForecastController::load_archive(json value, const std::string &path) {
  if constexpr (std::endian::native != std::endian::little)
    throw std::runtime_error(
        "This trajectory archive requires a little-endian host");
  const double issue = timestamp(value.at("issue_time"));
  const auto &animals = value.at("animals"), &days = value.at("times_days");
  const auto &shape = value.at("positions").at("shape");
  const auto count = animals.size(), steps = days.size();
  if (!animals.is_array() || !days.is_array() || !count || steps < 2 ||
      shape != json::array({steps, count, 2}) ||
      value["positions"].at("dtype") != "float32-le")
    throw std::runtime_error("Invalid trajectory archive dimensions");
  double last = -1;
  for (const auto &day : days) {
    const double time = day;
    if (!std::isfinite(time) || time <= last || time < 0)
      throw std::runtime_error(
          "Trajectory archive times must be finite and strictly ordered");
    last = time;
  }
  if (days.front() != 0 ||
      last != value.at("forecast_end_days").get<double>() || last > 36600)
    throw std::runtime_error("Trajectory archive time coverage is invalid");
  auto binary =
      std::filesystem::path(value["positions"].at("file").get<std::string>());
  if (binary.is_relative())
    binary = std::filesystem::path(path).parent_path() / binary;
  binary = std::filesystem::absolute(binary);
  if (count > 1000000 || steps > 200000 ||
      count * steps >
          std::numeric_limits<std::size_t>::max() / (2 * sizeof(float)) ||
      std::filesystem::file_size(binary) != count * steps * 2 * sizeof(float))
    throw std::runtime_error(
        "Trajectory binary length does not match its manifest");
  std::ifstream data(binary, std::ios::binary);
  if (!data)
    throw std::runtime_error("Cannot read trajectory positions");
  std::vector<TrackedAnimal> tracks;
  tracks.reserve(count);
  std::set<std::string> ids;
  std::size_t unknown = 0, limited = 0;
  for (const auto &animal : animals) {
    TrackedAnimal track;
    track.id = animal.at("id");
    track.species = animal.at("species");
    if (track.id.empty() || track.species.empty() ||
        !ids.insert(track.id).second)
      throw std::runtime_error(
          "Archive animal identities must be unique and nonempty");
    track.name = animal.value("name", track.species);
    track.cohort_id = animal.value("cohort_id", track.species);
    track.cohort_label = animal.value("cohort_label", "Recorded study cohort");
    track.group = color_family(track.species);
    track.forecast_start_hour = 0;
    track.hours.reserve(steps);
    track.positions.reserve(steps);
    tracks.push_back(std::move(track));
    unknown += animal.value("unknown_species_fallback", false);
    limited += !animal.value("history_support_matches_training", true);
  }
  std::vector<float> coordinates(count * 2);
  for (std::size_t time = 0; time < steps; ++time) {
    if (!data.read(reinterpret_cast<char *>(coordinates.data()),
                   coordinates.size() * sizeof(float)))
      throw std::runtime_error("Trajectory archive is truncated");
    for (std::size_t i = 0; i < count; ++i) {
      const float lat = coordinates[2 * i], lon = coordinates[2 * i + 1];
      if (!std::isfinite(lat) || !std::isfinite(lon) || std::abs(lat) > 90 ||
          std::abs(lon) > 180)
        throw std::runtime_error(
            "Trajectory archive contains coordinates outside Earth");
      tracks[i].hours.push_back(days[time].get<double>() * 24);
      tracks[i].positions.push_back(earth_position(lat, lon));
    }
  }
  value["positions"]["file"] = binary.string();
  value["tracks"] = json::array();
  value["imported_archive"] = true;
  result_ = std::move(value);
  imported_tracks_ = std::move(tracks);
  origin_seconds_ = issue;
  end_days_ = last;
  unknown_species_ = unknown;
  limited_history_ = limited;
  next_scenario_ = nullptr;
  continuing_ = false;
  mode_ = 1;
  apply_ = true;
  copy(observations_, "");
  copy(issue_time_, result_["issue_time"]);
  status_ = "Imported server model scenario: " + std::to_string(count) +
            " recorded animals / " + std::to_string(result_.value("years", 0)) +
            " years";
}
std::optional<AnimalId>
ForecastController::apply(Simulation &simulation,
                          std::vector<std::uint8_t> &enabled) {
  if (!apply_)
    return {};
  std::vector<TrackedAnimal> tracks = std::move(imported_tracks_);
  for (const auto &row : result_["tracks"]) {
    TrackedAnimal t;
    t.id = row.at("id");
    t.species = row.at("species");
    t.name = row.value("name", t.species);
    t.cohort_id = row.value("cohort_id", t.species);
    t.cohort_label = row.value("cohort_label", "Observed study cohort");
    t.group = color_family(t.species);
    for (const auto &fix :
         row.value(scenario() ? "scenario_history" : "observed_history",
                   json::array())) {
      t.hours.push_back((timestamp(fix.at("timestamp")) - origin_seconds_) /
                        3600);
      t.positions.push_back(earth_position(fix["lat"], fix["lon"]));
    }
    const auto &origin = row.at("origin");
    const double origin_hour =
        (timestamp(origin.at("timestamp")) - origin_seconds_) / 3600;
    if (t.hours.empty() || t.hours.back() < origin_hour) {
      t.hours.push_back(origin_hour);
      t.positions.push_back(earth_position(origin["lat"], origin["lon"]));
    }
    if (predicted()) {
      t.forecast_start_hour = origin_hour;
      if (scenario_generation() > 0 && !t.hours.empty())
        t.forecast_start_hour = t.hours.front();
      const auto sample = row.value("representative_sample", std::size_t(0));
      const auto &path = row["sampled_latlon_paths"][sample];
      for (std::size_t i = 0; i < path.size(); ++i) {
        t.hours.push_back(origin_hour +
                          row["horizons_days"][i].get<double>() * 24);
        t.positions.push_back(coordinate(path[i]));
      }
    }
    tracks.push_back(std::move(t));
  }
  std::vector<Species> catalog;
  if (result_.contains("dataset"))
    for (const auto &name :
         result_["dataset"].value("species_catalog", json::array())) {
      const auto taxon = name.get<std::string>();
      catalog.push_back(
          {taxon, Behavior::Herd, color_family(taxon), taxon, {}});
    }
  if (!simulation.replace_tracks(std::move(tracks), predicted(),
                                 std::move(catalog)))
    throw std::runtime_error("Invalid observed animal timelines");
  if (!continuing_ || enabled.size() != simulation.species().size())
    enabled.assign(simulation.species().size(), 1);
  apply_ = false;
  if (continuing_) {
    simulation.seek(resume_hours_);
    continuing_ = false;
    return {};
  }
  if (const auto selected = simulation.find_animal(selected_id_))
    return selected;
  if (!simulation.colonies().empty())
    return AnimalId{0, 0};
  return {};
}
void ForecastController::select(const Simulation &simulation,
                                std::optional<AnimalId> id) {
  selected_id_ = id ? std::string(simulation.animal_name(*id)) : std::string{};
}
std::optional<ForecastClock> ForecastController::clock(double hours) const {
  if (!ready())
    return {};
  using namespace std::chrono;
  const auto now = sys_seconds{seconds{static_cast<long long>(
      origin_seconds_ + std::clamp(hours, 0., end_days_ * 24) * 3600)}};
  const year_month_day date{floor<days>(now)};
  const auto begin = sys_days{date.year() / January / 1},
             end = sys_days{(date.year() + years{1}) / January / 1};
  const double begin_seconds =
                   duration<double>(begin.time_since_epoch()).count(),
               year_hours = duration<double>(end - begin).count() / 3600;
  return ForecastClock{
      int(date.year()), duration<double>(now - begin).count() / 3600,
      std::clamp((origin_seconds_ - begin_seconds) / 3600, 0., year_hours),
      std::clamp((origin_seconds_ - begin_seconds) / 3600 + end_days_ * 24, 0.,
                 year_hours)};
}
void ForecastController::seek_relative(Simulation &simulation,
                                       double delta) const {
  if (ready())
    simulation.seek(std::clamp(
        simulation.elapsed_hours() + delta, 0.,
        (scenario() ? result_["scenario"].value("step_days", 30.) : end_days_) *
            24));
}
std::optional<AnimalId>
ForecastController::controls(Simulation &simulation,
                             std::vector<std::uint8_t> &enabled) {
  auto focus = apply(simulation, enabled);
  ImGui::SeparatorText("WildCast cohort forecast");
  ImGui::TextWrapped("%s", status_.c_str());
  if (!health_.is_null()) {
    const std::string gpu =
        health_.contains("gpu") && health_["gpu"].is_string()
            ? health_["gpu"].get<std::string>()
            : "CPU";
    ImGui::TextWrapped("%s | %s", health_.value("device", "").c_str(),
                       gpu.c_str());
    ImGui::Text("Model: %d trained taxon labels",
                health_.value("trained_taxa", 0));
  }
  if (ready() && result_.contains("dataset")) {
    const auto &d = result_["dataset"];
    ImGui::Text("GPS archive: %d taxa / %d animals",
                d.value("recorded_species", 0), d.value("recorded_animals", 0));
    ImGui::Text("At issue: %zu animals / %zu cohorts",
                simulation.animal_count(), simulation.colonies().size());
    if (unknown_species_) {
      ImGui::TextColored({.95f, .72f, .35f, 1},
                         "%zu animals: untrained taxon label",
                         unknown_species_);
      ImGui::SetItemTooltip(
          "These real animals use the checkpoint's unknown-species token.");
    }
    if (limited_history_) {
      ImGui::TextColored({.95f, .72f, .35f, 1},
                         "%zu animals: limited GPS history", limited_history_);
      ImGui::SetItemTooltip("Their available GPS history is below the "
                            "checkpoint's training support thresholds.");
    }
  }
  ImGui::TextUnformatted("Observed GPS dataset");
  ImGui::SetNextItemWidth(-1);
  ImGui::InputText("##observations", observations_, sizeof(observations_));
  ImGui::BeginDisabled(pending());
  ImGui::SetNextItemWidth(-1);
  ImGui::Combo("##mode", &mode_,
               "Continuous model scenario\0Historical forecast\0");
  ImGui::EndDisabled();
  ImGui::TextUnformatted(mode_ == 0 ? "Scenario start (UTC; empty = now)"
                                    : "Forecast issue time (UTC)");
  ImGui::SetNextItemWidth(-1);
  ImGui::InputText("##issue", issue_time_, sizeof(issue_time_));
  if (ImGui::TreeNode("Forecast settings")) {
    ImGui::SetNextItemWidth(90);
    ImGui::SliderInt("Uncertainty paths", &samples_, 2, 64);
    if (mode_ == 1) {
      ImGui::SetNextItemWidth(90);
      ImGui::SliderInt("GPS freshness (hours)", &max_age_hours_, 1, 168);
      ImGui::SetItemTooltip("Only actual GPS fixes within this interval before "
                            "the issue time are eligible.");
    }
    ImGui::TreePop();
  }
  ImGui::BeginDisabled(pending());
  if (ImGui::Button(mode_ == 0 ? "Start new model scenario"
                               : "Forecast all observed animals")) {
    if (mode_ == 0)
      start_scenario(false);
    else
      start_forecast();
  }
  ImGui::EndDisabled();
  if (ImGui::TreeNode("Forecast file")) {
    ImGui::SetNextItemWidth(-1);
    ImGui::InputText("##json", json_path_, sizeof(json_path_));
    if (ImGui::SmallButton("Load JSON"))
      load(json_path_);
    ImGui::SameLine();
    ImGui::BeginDisabled(!ready());
    if (ImGui::SmallButton("Save JSON")) {
      try {
        const auto path = std::filesystem::path(json_path_);
        if (path.has_parent_path())
          std::filesystem::create_directories(path.parent_path());
        std::ofstream file(path);
        file << result_.dump();
        if (!file)
          throw std::runtime_error("Could not save forecast");
        status_ = "Saved " + path.string();
      } catch (const std::exception &error) {
        status_ = error.what();
      }
    }
    ImGui::EndDisabled();
    ImGui::TreePop();
  }
  if (ready()) {
    if (result_.value("imported_archive", false)) {
      ImGui::TextColored({.5f, .8f, 1, 1},
                         "Imported server ensemble | %d years",
                         result_.value("years", 0));
      ImGui::TextWrapped("Recorded identities with precomputed model "
                         "trajectories. Playback makes no inference requests.");
    }
    if (scenario()) {
      ImGui::TextColored({.5f, .8f, 1, 1},
                         "Continuous model scenario | cycle %d",
                         scenario_generation() + 1);
      ImGui::Text("Projected +%.1f days",
                  result_["scenario"].value("elapsed_days", 0.) +
                      simulation.elapsed_hours() / 24);
      ImGui::TextWrapped("Historical last-known positions initialize this "
                         "scenario. Predicted movement feeds later forecasts; "
                         "these are not live animal locations.");
    }
    if (result_.value("retrospective_replay", false))
      ImGui::TextColored({.95f, .72f, .35f, 1}, "Historical forecast replay");
    float day = float(simulation.elapsed_hours() / 24);
    ImGui::SetNextItemWidth(-1);
    ImGui::BeginDisabled(!predicted());
    if (ImGui::SliderFloat(
            "##day", &day, 0,
            float(scenario() ? result_["scenario"].value("step_days", 30.)
                             : end_days_),
            "Forecast +%.2f days"))
      simulation.seek(day * 24);
    ImGui::EndDisabled();
    ImGui::Checkbox("Selected animal: uncertainty", &show_paths_);
    ImGui::Checkbox("Selected animal: GPS history", &show_history_);
    ImGui::TextWrapped("Cohorts group study and species. Each particle is a "
                       "recorded animal; alternative paths are "
                       "uncertainty, not extra animals.");
  }
  return focus;
}
void ForecastController::draw(ImDrawList &draw, const GlobeView &globe,
                              const Simulation &) const {
  if (!ready() || selected_id_.empty())
    return;
  const auto &rows = result_["tracks"];
  const auto found = std::find_if(rows.begin(), rows.end(), [&](const json &t) {
    return t.at("id") == selected_id_;
  });
  if (found == rows.end())
    return;
  const auto &t = *found;
  const auto project = [&](Vec3 p, ImVec2 &at) { return globe.project(p, at); };
  if (show_history_) {
    std::optional<Vec3> last;
    for (const auto &row : t.value("observed_history", json::array())) {
      const auto p = earth_position(row["lat"], row["lon"]);
      if (last)
        line(draw, project, *last, p, IM_COL32(207, 224, 235, 145), 1.2f);
      last = p;
    }
  }
  if (show_paths_ && predicted()) {
    const auto start = earth_position(t["origin"]["lat"], t["origin"]["lon"]);
    const auto selected = t.value("representative_sample", std::size_t(0));
    const auto &paths = t["sampled_latlon_paths"];
    for (std::size_t i = 0; i < paths.size(); ++i) {
      auto a = start;
      for (const auto &row : paths[i]) {
        const auto b = coordinate(row);
        line(draw, project, a, b,
             i == selected ? IM_COL32(255, 208, 102, 210)
                           : IM_COL32(105, 211, 217, 35),
             i == selected ? 1.6f : .8f);
        a = b;
      }
    }
  }
}
} // namespace wildlife
