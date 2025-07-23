import React from 'react';
import AgentCard from '../components/AgentCard';
import WeightHistogram from '../components/WeightHistogram';
import { useNavigate } from 'react-router-dom';

const agents = [
  {
    key: 'training',
    name: 'Training Agent',
    description: 'Personalized fitness plans and workout tracking.',
    icon: '🏋️',
    route: '/agents/training',
    color: 'from-pink-500 to-red-400',
  },
  {
    key: 'diet',
    name: 'Diet Agent',
    description: 'Personalized nutrition plans and meal logging.',
    icon: '🥗',
    route: '/agents/diet',
    color: 'from-green-400 to-lime-400',
  },
  {
    key: 'habit',
    name: 'Habit Agent',
    description: 'Habit formation, tracking, and suggestions.',
    icon: '📈',
    route: '/agents/habit',
    color: 'from-blue-400 to-cyan-400',
  },
  {
    key: 'qol',
    name: 'QoL Agent',
    description: 'Quality of life assessment and recommendations.',
    icon: '🌟',
    route: '/agents/qol',
    color: 'from-yellow-400 to-orange-400',
  },
  {
    key: 'orchestrator',
    name: 'Orchestrator',
    description: 'Aggregates all agents for holistic coaching.',
    icon: '🤖',
    route: '/orchestrator',
    color: 'from-purple-400 to-fuchsia-400',
  },
];

export default function Dashboard() {
  const navigate = useNavigate();
  return (
    <div className="py-8 px-4 max-w-6xl mx-auto">
      <h1 className="text-3xl font-bold mb-6">Dashboard</h1>
      <div className="mb-8">
        <WeightHistogram />
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        {agents.map(agent => (
          <AgentCard
            key={agent.key}
            name={agent.name}
            description={agent.description}
            icon={agent.icon}
            color={agent.color}
            onClick={() => navigate(agent.route)}
          />
        ))}
      </div>
    </div>
  );
}
