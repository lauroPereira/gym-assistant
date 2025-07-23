import React from 'react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

// Example weight data (date, weight in kg)
const sampleData = [
  { date: '2025-07-01', weight: 85 },
  { date: '2025-07-05', weight: 84.5 },
  { date: '2025-07-10', weight: 83.8 },
  { date: '2025-07-15', weight: 83.2 },
  { date: '2025-07-20', weight: 82.7 },
  { date: '2025-07-22', weight: 82.3 },
];

export default function WeightHistogram({ data = sampleData }) {
  return (
    <div className="bg-white rounded-xl shadow p-6 w-full">
      <h2 className="text-lg font-semibold mb-4">Weight Loss Over Time</h2>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data} margin={{ top: 16, right: 16, left: 0, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" tick={{ fontSize: 12 }} />
          <YAxis label={{ value: 'Weight (kg)', angle: -90, position: 'insideLeft', fontSize: 12 }} />
          <Tooltip />
          <Legend />
          <Bar dataKey="weight" fill="#3b82f6" name="Weight (kg)" barSize={32} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
