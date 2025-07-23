import React, { useState, useEffect } from 'react';
import LoadingSpinner from '../components/LoadingSpinner';

const Habits = () => {
  const [habits, setHabits] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchHabits();
  }, []);

  const fetchHabits = async () => {
    try {
      setTimeout(() => {
        setHabits([
          { id: 1, name: 'Beber 2L de água', completed: true },
          { id: 2, name: 'Dormir 8h', completed: false },
          { id: 3, name: 'Caminhar 7000 passos', completed: true },
          { id: 4, name: 'Meditar 10 minutos', completed: false }
        ]);
        setLoading(false);
      }, 800);
    } catch (err) {
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="flex justify-center items-center h-64"><LoadingSpinner size="large" /></div>;
  }

  return (
    <div className="max-w-xl mx-auto p-6">
      <h1 className="text-3xl font-bold mb-4 text-gray-900">Hábitos</h1>
      <ul className="space-y-4">
        {habits.map(habit => (
          <li key={habit.id} className={`flex items-center justify-between bg-white rounded-lg shadow p-4 ${habit.completed ? 'border-green-400 border-l-4' : 'border-gray-200 border-l-4'}`}>
            <span>{habit.name}</span>
            <span className={`px-2 py-1 rounded text-xs font-semibold ${habit.completed ? 'bg-green-100 text-green-800' : 'bg-gray-200 text-gray-800'}`}>{habit.completed ? 'Feito' : 'Pendente'}</span>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default Habits;
