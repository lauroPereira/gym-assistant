import React from 'react';

export default function AgentCard({ name, description, icon, color, onClick }) {
  return (
    <div
      className={`bg-gradient-to-br ${color} rounded-xl shadow-lg p-6 cursor-pointer hover:scale-105 transition-transform duration-200 flex flex-col justify-between min-h-[180px]`}
      onClick={onClick}
      role="button"
      tabIndex={0}
      onKeyPress={e => { if (e.key === 'Enter') onClick(); }}
      aria-label={`View details for ${name}`}
    >
      <div className="flex items-center gap-4 mb-4">
        <span className="text-4xl" aria-hidden>{icon}</span>
        <h2 className="text-xl font-semibold text-white drop-shadow">{name}</h2>
      </div>
      <p className="text-white/90 mb-2">{description}</p>
      <span className="text-sm text-white/70">Click for details &rarr;</span>
    </div>
  );
}
