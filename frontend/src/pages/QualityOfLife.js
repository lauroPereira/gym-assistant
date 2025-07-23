import React, { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import LoadingSpinner from '../components/LoadingSpinner';

const QualityOfLife = () => {
  const { user } = useAuth();
  const [metrics, setMetrics] = useState([]);
  const [loading, setLoading] = useState(true);
  const [recommendations, setRecommendations] = useState([]);

  useEffect(() => {
    fetchMetrics();
    fetchRecommendations();
  }, []);

  const fetchMetrics = async () => {
    try {
      // Simulated metrics data
      setTimeout(() => {
        setMetrics([
          { id: 1, name: "Sono", value: 7.5, unit: "h", status: "Bom" },
          { id: 2, name: "Estresse", value: 3, unit: "/10", status: "Baixo" },
          { id: 3, name: "Bem-estar", value: 8, unit: "/10", status: "Alto" },
          { id: 4, name: "Passos", value: 9000, unit: "passos", status: "Ótimo" }
        ]);
        setLoading(false);
      }, 1000);
    } catch (error) {
      console.error('Erro ao buscar métricas:', error);
      setLoading(false);
    }
  };

  const fetchRecommendations = async () => {
    try {
      setRecommendations([
        "Mantenha uma rotina de sono regular.",
        "Faça pausas para relaxamento durante o dia.",
        "Pratique atividades físicas leves diariamente.",
        "Busque momentos de lazer e socialização."
      ]);
    } catch (error) {
      console.error('Erro ao buscar recomendações:', error);
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <LoadingSpinner size="large" />
      </div>
    );
  }

  return (
    <div className="max-w-3xl mx-auto p-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Qualidade de Vida</h1>
        <p className="text-gray-600">Acompanhe suas métricas e recomendações para bem-estar</p>
      </div>
      <div className="bg-white rounded-lg shadow-md p-6 mb-8">
        <h2 className="text-2xl font-semibold mb-4">Métricas Recentes</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {metrics.map((metric) => (
            <div key={metric.id} className="border border-gray-200 rounded-lg p-4 flex flex-col justify-between">
              <div>
                <h3 className="font-semibold text-lg mb-1">{metric.name}</h3>
                <span className="text-2xl font-bold text-blue-700">{metric.value} {metric.unit}</span>
              </div>
              <span className="mt-2 px-2 py-1 rounded text-xs font-semibold bg-green-100 text-green-800 w-max">{metric.status}</span>
            </div>
          ))}
        </div>
      </div>
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-2xl font-semibold mb-4">Recomendações</h2>
        <ul className="list-disc pl-5 text-gray-700">
          {recommendations.map((rec, idx) => (
            <li key={idx} className="mb-2">{rec}</li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default QualityOfLife;
