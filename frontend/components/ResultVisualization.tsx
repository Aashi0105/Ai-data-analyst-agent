"use client";

import React from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  LineChart,
  Line,
  ScatterChart,
  Scatter,
  PieChart,
  Pie,
  Cell,
  Legend,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";
import { VisualizationData } from "@/types";

interface ResultVisualizationProps {
  visualization: VisualizationData;
}

export const ResultVisualization: React.FC<ResultVisualizationProps> = ({ visualization }) => {
  const { chart_type, data, x_key, y_keys, title } = visualization;

  if (!data || data.length === 0) {
    return null;
  }

  const primaryColor = "#2D5A43"; // Muted forest green
  const axisColor = "#78716C";
  const gridColor = "#E7E5E0";

  // Refined editorial palette for categorical & pie slices
  const sliceColors = [
    "#2D5A43", // editorial.accent (forest green)
    "#4A7C59", // editorial.sage
    "#4A3F55", // editorial.plum
    "#9C6615", // editorial.ochre
    "#607D8B", // slate grey
    "#8D6E63", // warm taupe
    "#3F51B5", // muted indigo
    "#00796B", // deep teal
  ];

  return (
    <div className="w-full mt-6 pt-5 border-t border-editorial-border">
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-xs font-semibold uppercase tracking-wider text-charcoal-subtle">
          {title || `${chart_type}`}
        </h4>
        <span className="text-[10px] text-charcoal-muted px-2 py-0.5 rounded border border-editorial-border bg-surface">
          {chart_type}
        </span>
      </div>

      <div className="w-full h-72 sm:h-80 bg-surface border border-editorial-border rounded-lg p-4">
        <ResponsiveContainer width="100%" height="100%">
          {chart_type === "Pie Chart" ? (
            <PieChart margin={{ top: 10, right: 10, left: 10, bottom: 10 }}>
              <Tooltip
                formatter={(val: any) => [
                  typeof val === "number" && val < 1 && val > 0
                    ? `${(val * 100).toFixed(1)}%`
                    : val,
                  "Value"
                ]}
                contentStyle={{
                  backgroundColor: "#FFFFFF",
                  border: "1px solid #E7E5E0",
                  borderRadius: "6px",
                  fontSize: "12px",
                  boxShadow: "0 2px 8px rgba(0,0,0,0.04)",
                }}
              />
              <Legend
                verticalAlign="bottom"
                height={36}
                iconType="circle"
                wrapperStyle={{ fontSize: "11px", color: axisColor, paddingTop: "8px" }}
              />
              <Pie
                data={data}
                dataKey={y_keys[0] || "value"}
                nameKey={x_key || "name"}
                cx="50%"
                cy="45%"
                outerRadius={85}
                innerRadius={36}
                paddingAngle={2}
                label={({ name, percent }: any) =>
                  percent !== undefined && percent > 0.05
                    ? `${name} (${(percent * 100).toFixed(0)}%)`
                    : ""
                }
                labelLine={false}
              >
                {data.map((_, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={sliceColors[index % sliceColors.length]}
                    stroke="#FFFFFF"
                    strokeWidth={1.5}
                  />
                ))}
              </Pie>
            </PieChart>
          ) : chart_type === "Bar Chart" || chart_type === "Histogram" ? (
            <BarChart data={data} margin={{ top: 10, right: 15, left: 10, bottom: 25 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={gridColor} vertical={false} />
              <XAxis
                dataKey={x_key}
                stroke={axisColor}
                fontSize={11}
                tickLine={false}
                interval={data.length > 10 ? 1 : 0}
                angle={data.length > 5 ? -25 : 0}
                textAnchor={data.length > 5 ? "end" : "middle"}
              />
              <YAxis
                stroke={axisColor}
                fontSize={11}
                tickLine={false}
                axisLine={false}
                tickFormatter={(val) => (typeof val === "number" && val >= 1000 ? `${(val / 1000).toFixed(1)}k` : val)}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#FFFFFF",
                  border: "1px solid #E7E5E0",
                  borderRadius: "6px",
                  fontSize: "12px",
                  boxShadow: "0 2px 8px rgba(0,0,0,0.04)",
                }}
              />
              <Bar dataKey={y_keys[0] || "value"} fill={primaryColor} radius={[3, 3, 0, 0]} />
            </BarChart>
          ) : chart_type === "Line Chart" ? (
            <LineChart data={data} margin={{ top: 10, right: 15, left: 10, bottom: 25 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={gridColor} vertical={false} />
              <XAxis
                dataKey={x_key}
                stroke={axisColor}
                fontSize={11}
                tickLine={false}
                angle={-25}
                textAnchor="end"
              />
              <YAxis
                stroke={axisColor}
                fontSize={11}
                tickLine={false}
                axisLine={false}
                tickFormatter={(val) => (typeof val === "number" && val >= 1000 ? `${(val / 1000).toFixed(1)}k` : val)}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#FFFFFF",
                  border: "1px solid #E7E5E0",
                  borderRadius: "6px",
                  fontSize: "12px",
                  boxShadow: "0 2px 8px rgba(0,0,0,0.04)",
                }}
              />
              <Line
                type="monotone"
                dataKey={y_keys[0] || "value"}
                stroke={primaryColor}
                strokeWidth={2}
                dot={{ r: 3, fill: primaryColor }}
                activeDot={{ r: 5 }}
              />
            </LineChart>
          ) : chart_type === "Scatter Plot" ? (
            <ScatterChart margin={{ top: 10, right: 15, left: 10, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
              <XAxis dataKey="x" name={x_key} stroke={axisColor} fontSize={11} tickLine={false} />
              <YAxis dataKey="y" name={y_keys[0]} stroke={axisColor} fontSize={11} tickLine={false} axisLine={false} />
              <Tooltip
                cursor={{ strokeDasharray: "3 3" }}
                contentStyle={{
                  backgroundColor: "#FFFFFF",
                  border: "1px solid #E7E5E0",
                  borderRadius: "6px",
                  fontSize: "12px",
                }}
              />
              <Scatter data={data} fill={primaryColor} />
            </ScatterChart>
          ) : (
            <div className="flex items-center justify-center h-full text-xs text-charcoal-muted">
              Chart type &quot;{chart_type}&quot; preview ready
            </div>
          )}
        </ResponsiveContainer>
      </div>
    </div>
  );
};
