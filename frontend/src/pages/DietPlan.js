import React, { useState, useEffect } from 'react';
import LoadingSpinner from '../components/LoadingSpinner';

const DietPlan = () => {
  const [plan, setPlan] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchPlan();
  }, []);

  const fetchPlan = async () => {
    try {
      setTimeout(() => {
        setPlan({
          name: 'Plano Alimentar - Hipertrofia',
          description: 'Plano alimentar focado em ganho de massa muscular',
          meals: [
            { time: 'Café da manhã', items: ['Ovos mexidos', 'Aveia', 'Banana', 'Café preto'] },
            { time: 'Almoço', items: ['Arroz', 'Frango grelhado', 'Brócolis', 'Batata doce'] },
            { time: 'Lanche da tarde', items: ['Iogurte', 'Granola', 'Maçã'] },
            { time: 'Jantar', items: ['Peixe', 'Quinoa', 'Salada verde'] }
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
      <div className="space-y-6">
        {plan.meals.map((meal, idx) => (
          <div key={idx} className="bg-white rounded-lg shadow p-4">
            <h2 className="text-lg font-semibold mb-2">{meal.time}</h2>
            <ul className="list-disc pl-5">
              {meal.items.map((item, i) => (
                <li key={i}>{item}</li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </div>
  );
};

export default DietPlan;
