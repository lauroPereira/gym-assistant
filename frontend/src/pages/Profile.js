import React, { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import LoadingSpinner from '../components/LoadingSpinner';

const Profile = () => {
  const { user } = useAuth();
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);
  const [editMode, setEditMode] = useState(false);
  const [form, setForm] = useState({});

  useEffect(() => {
    fetchProfile();
  }, []);

  const fetchProfile = async () => {
    try {
      // Simulated user profile data
      setTimeout(() => {
        const data = {
          name: "João Silva",
          email: "joao.silva@email.com",
          age: 32,
          gender: "Masculino",
          goal: "Ganhar massa muscular",
          preferences: {
            notifications: true,
            darkMode: false
          }
        };
        setProfile(data);
        setForm(data);
        setLoading(false);
      }, 800);
    } catch (error) {
      setLoading(false);
    }
  };

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    if (name === "notifications" || name === "darkMode") {
      setForm({
        ...form,
        preferences: {
          ...form.preferences,
          [name]: type === "checkbox" ? checked : value
        }
      });
    } else {
      setForm({ ...form, [name]: value });
    }
  };

  const handleSave = () => {
    setProfile(form);
    setEditMode(false);
    // Aqui você pode integrar com API para salvar alterações
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <LoadingSpinner size="large" />
      </div>
    );
  }

  return (
    <div className="max-w-xl mx-auto p-6">
      <h1 className="text-3xl font-bold mb-4 text-gray-900">Perfil do Usuário</h1>
      <div className="bg-white rounded-lg shadow-md p-6">
        {editMode ? (
          <form className="space-y-4">
            <div>
              <label className="block text-gray-700">Nome</label>
              <input type="text" name="name" value={form.name} onChange={handleChange} className="input" />
            </div>
            <div>
              <label className="block text-gray-700">Email</label>
              <input type="email" name="email" value={form.email} onChange={handleChange} className="input" />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-gray-700">Idade</label>
                <input type="number" name="age" value={form.age} onChange={handleChange} className="input" />
              </div>
              <div>
                <label className="block text-gray-700">Gênero</label>
                <select name="gender" value={form.gender} onChange={handleChange} className="input">
                  <option value="Masculino">Masculino</option>
                  <option value="Feminino">Feminino</option>
                  <option value="Outro">Outro</option>
                </select>
              </div>
            </div>
            <div>
              <label className="block text-gray-700">Objetivo</label>
              <input type="text" name="goal" value={form.goal} onChange={handleChange} className="input" />
            </div>
            <div className="flex items-center">
              <input type="checkbox" name="notifications" checked={form.preferences.notifications} onChange={handleChange} className="mr-2" />
              <span>Receber notificações</span>
            </div>
            <div className="flex items-center">
              <input type="checkbox" name="darkMode" checked={form.preferences.darkMode} onChange={handleChange} className="mr-2" />
              <span>Modo escuro</span>
            </div>
            <div className="flex gap-2 mt-4">
              <button type="button" className="btn-primary" onClick={handleSave}>Salvar</button>
              <button type="button" className="btn-secondary" onClick={() => setEditMode(false)}>Cancelar</button>
            </div>
          </form>
        ) : (
          <div className="space-y-4">
            <div><span className="font-semibold">Nome:</span> {profile.name}</div>
            <div><span className="font-semibold">Email:</span> {profile.email}</div>
            <div><span className="font-semibold">Idade:</span> {profile.age}</div>
            <div><span className="font-semibold">Gênero:</span> {profile.gender}</div>
            <div><span className="font-semibold">Objetivo:</span> {profile.goal}</div>
            <div><span className="font-semibold">Notificações:</span> {profile.preferences.notifications ? 'Ativado' : 'Desativado'}</div>
            <div><span className="font-semibold">Modo escuro:</span> {profile.preferences.darkMode ? 'Ativado' : 'Desativado'}</div>
            <button className="btn-primary mt-4" onClick={() => setEditMode(true)}>Editar Perfil</button>
          </div>
        )}
      </div>
    </div>
  );
};

export default Profile;
