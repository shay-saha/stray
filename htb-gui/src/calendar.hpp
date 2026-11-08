#pragma once

#include <imgui.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <optional>

namespace wildlife {

struct CalendarRecording {
    int year;
    double begin_hours;
    double end_hours;
};

namespace calendar_detail {

inline constexpr std::array month_starts{0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365};
inline constexpr std::array months{"JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                                   "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"};
inline constexpr float tau = 2.0f * std::numbers::pi_v<float>;
inline constexpr float top = -std::numbers::pi_v<float> * 0.5f;

struct Date {
    double year;
    std::size_t month;
    int day;
    int minute_of_day;
    float year_progress;
    float day_progress;
};

inline bool leap_year(int year) {
    return year != 0 && year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
}

inline int month_start_day(std::size_t month, int year) {
    return month_starts[month] + (month >= 2 && leap_year(year) ? 1 : 0);
}

inline int year_hours(int year) {
    return month_start_day(12, year) * 24;
}

inline Date read_date(double hours, int calendar_year = 0) {
    const double elapsed = std::isfinite(hours) && hours >= 0.0 ? hours : 0.0;
    const int duration = year_hours(calendar_year);
    const double within_year = std::fmod(elapsed, duration);
    constexpr int minutes_per_day = 24 * 60;
    const int year_minutes = std::min(static_cast<int>(within_year * 60.0), duration * 60 - 1);
    const int day_of_year = year_minutes / minutes_per_day;
    std::size_t month = 0;
    while (month_start_day(month + 1, calendar_year) <= day_of_year) ++month;
    return {1.0 + std::floor(elapsed / duration), month, day_of_year - month_start_day(month, calendar_year) + 1,
            year_minutes % minutes_per_day, static_cast<float>(within_year / duration),
            static_cast<float>(std::fmod(within_year, 24.0) / 24.0)};
}

inline ImVec2 radial(ImVec2 center, float radius, float angle) {
    return {center.x + radius * std::cos(angle), center.y + radius * std::sin(angle)};
}

inline void center_text(ImDrawList& draw, ImVec2 center, float size, ImU32 color, const char* text) {
    ImFont* font = ImGui::GetFont();
    const ImVec2 bounds = font->CalcTextSizeA(size, FLT_MAX, 0.0f, text);
    draw.AddText(font, size, {std::round(center.x - bounds.x * 0.5f),
                              std::round(center.y - bounds.y * 0.5f)}, color, text);
}

} // namespace calendar_detail

inline std::optional<double> calendar_scrub_hours(ImVec2 center, float radius, ImVec2 cursor,
                                                  double elapsed_hours, bool dragging, int calendar_year = 0) {
    const float dx = cursor.x - center.x;
    const float dy = cursor.y - center.y;
    const float distance_squared = dx * dx + dy * dy;
    const float inner = radius * 0.56f;
    const float outer = radius * 1.02f;
    if (!std::isfinite(distance_squared) || distance_squared < 1.0f ||
        (!dragging && (distance_squared < inner * inner || distance_squared > outer * outer))) {
        return std::nullopt;
    }

    constexpr double full_turn = 2.0 * std::numbers::pi;
    const double year = calendar_detail::year_hours(calendar_year);
    const double angle = std::atan2(static_cast<double>(dx), -static_cast<double>(dy));
    const double phase = (angle < 0.0 ? angle + full_turn : angle) / full_turn;
    const double elapsed = std::isfinite(elapsed_hours) && elapsed_hours >= 0.0 ? elapsed_hours : 0.0;
    return (std::floor(elapsed / year) + phase) * year;
}

// Both dials advance clockwise from January 1 and midnight respectively.
inline void draw_calendar(ImDrawList& draw, ImVec2 center, float radius, double elapsed_hours,
                          std::optional<CalendarRecording> recording = std::nullopt) {
    using namespace calendar_detail;
    const int calendar_year = recording ? recording->year : 0;
    const int duration = year_hours(calendar_year);
    const int days = duration / 24;
    const Date date = read_date(elapsed_hours, calendar_year);
    constexpr ImU32 ink = IM_COL32(241, 235, 214, 255);
    constexpr ImU32 muted_ink = IM_COL32(154, 181, 176, 255);
    constexpr ImU32 accent = IM_COL32(157, 226, 201, 255);
    constexpr ImU32 rail = IM_COL32(110, 152, 151, 76);
    constexpr std::array quarter_colors{IM_COL32(115, 160, 176, 210), IM_COL32(126, 171, 139, 210),
                                        IM_COL32(193, 166, 109, 210), IM_COL32(172, 140, 145, 210)};
    const float outer_track = radius * 0.91f;
    const float inner_track = radius * 0.53f;
    const float text_scale = std::clamp(radius / 90.0f, 0.72f, 1.25f);

    draw.AddCircleFilled({center.x, center.y + 2.0f}, radius + 2.0f, IM_COL32(3, 9, 11, 120), 64);
    draw.AddCircleFilled(center, radius, IM_COL32(11, 25, 29, 248), 64);
    draw.AddCircle(center, radius, IM_COL32(126, 167, 158, 108), 64);
    draw.AddCircleFilled(center, inner_track, IM_COL32(18, 38, 42, 255), 48);
    draw.AddCircle(center, inner_track, rail, 48);
    if (recording) {
        const float start = top + tau * static_cast<float>(recording->begin_hours / duration);
        const float end = top + tau * static_cast<float>(recording->end_hours / duration);
        draw.PathArcTo(center, radius * 0.97f, start, end, 24);
        draw.PathStroke(IM_COL32(157, 226, 201, 178), 0, 1.5f);
    }

    for (std::size_t month = 0; month < months.size(); ++month) {
        const int first_day = month_start_day(month, calendar_year);
        const int next_month = month_start_day(month + 1, calendar_year);
        const float month_start = top + tau * static_cast<float>(first_day) / days;
        const float month_end = top + tau * static_cast<float>(next_month) / days;
        const bool active = month == date.month;
        constexpr float gap = 0.017f;
        draw.PathArcTo(center, outer_track, month_start + gap, month_end - gap, 9);
        draw.PathStroke(active ? accent : quarter_colors[month / 3], 0, active ? 2.5f : 1.6f);
        draw.AddLine(radial(center, radius * 0.81f, month_start), radial(center, radius * 0.88f, month_start), rail);
        for (int day = first_day + 7; day < next_month; day += 7) {
            const float angle = top + tau * static_cast<float>(day) / days;
            draw.AddLine(radial(center, radius * 0.85f, angle), radial(center, radius * 0.88f, angle), rail);
        }
        const float midpoint = (month_start + month_end) * 0.5f;
        center_text(draw, radial(center, radius * 0.72f, midpoint), 11.0f * text_scale,
                    active ? ink : muted_ink, months[month]);
    }

    const float day_angle = top + tau * date.day_progress;
    if (date.day_progress > 0.0f) {
        draw.PathArcTo(center, inner_track, top, day_angle, 1 + static_cast<int>(date.day_progress * 48.0f));
        draw.PathStroke(IM_COL32(99, 191, 172, 175), 0, 1.4f);
    }
    draw.AddCircleFilled(radial(center, inner_track, day_angle), 2.2f * text_scale, accent, 8);

    const ImVec2 year_marker = radial(center, outer_track, top + tau * date.year_progress);
    draw.AddCircleFilled(year_marker, 4.6f * text_scale, IM_COL32(9, 19, 23, 255), 12);
    draw.AddCircleFilled(year_marker, 3.1f * text_scale, ink, 12);

    char date_time_label[24];
    char year_label[24];
    std::snprintf(date_time_label, sizeof(date_time_label), "%02d  /  %02d:%02d",
                  date.day, date.minute_of_day / 60, date.minute_of_day % 60);
    if (recording) {
        std::snprintf(year_label, sizeof(year_label), "%d UTC", recording->year);
    } else {
        std::snprintf(year_label, sizeof(year_label), "YEAR %.0f", date.year);
    }
    center_text(draw, {center.x, center.y - radius * 0.25f}, 10.0f * text_scale, muted_ink, year_label);
    center_text(draw, center, 26.0f * text_scale, ink, months[date.month]);
    center_text(draw, {center.x, center.y + radius * 0.25f}, 11.0f * text_scale, accent, date_time_label);
}

} // namespace wildlife
