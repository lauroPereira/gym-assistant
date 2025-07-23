import React, { useState, useEffect } from 'react';
import LoadingSpinner from '../components/LoadingSpinner';

const TrainingPlan = () => {
  const [plan, setPlan] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchPlan();
  }, []);

  const fetchPlan = async () => {
    try {
      setTimeout(() => {
        setPlan({
          name: 'Plano de Treino - Hipertrofia',
          description: 'Plano focado em ganho de massa muscular',
          days: [
            {
              day: 'Segunda-feira',
              exercises: [
                { name: 'Supino Reto', sets: 4, reps: 10 },
                { name: 'Crucifixo', sets: 3, reps: 12 },
                { name: 'Tríceps Testa', sets: 3, reps: 12 }
              ]
            },
            {
              day: 'Quarta-feira',
              exercises: [
                { name: 'Agachamento', sets: 4, reps: 10 },
                { name: 'Leg Press', sets: 3, reps: 12 },
                { name: 'Flexora', sets: 3, reps: 12 }
              ]
            }
          ]
        });
        setLoading(false);
      }, 1000);
    } catch (err) {
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="flex justify-center items-center h-64"><LoadingSpinner size="large" /></div>;
  }

  return (
    <div className="max-w-3xl mx-auto p-6">
      <h1 className="text-3xl font-bold mb-4 text-gray-900">{plan.name}</h1>
      <p className="mb-6 text-gray-700">{plan.description}</p>
      {plan.days.map((day, idx) => (
        <div key={idx} className="mb-6">
          <h2 className="text-xl font-semibold mb-2">{day.day}</h2>
          <table className="min-w-full bg-white border border-gray-200 rounded-lg">
            <thead>
              <tr>
                <th className="py-2 px-4 border-b">Exercício</th>
                <th className="py-2 px-4 border-b">Séries</th>
                <th className="py-2 px-4 border-b">Repetições</th>
              </tr>
            </thead>
            <tbody>
              {day.exercises.map((ex, i) => (
                <tr key={i}>
                  <td className="py-2 px-4 border-b">{ex.name}</td>
                  <td className="py-2 px-4 border-b text-center">{ex.sets}</td>
                  <td className="py-2 px-4 border-b text-center">{ex.reps}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
    </div>
  );
};

export default TrainingPlan;
