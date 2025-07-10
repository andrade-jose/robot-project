import React, { useState, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { TrendingUp, TrendingDown, Target, Zap, Award, AlertCircle } from 'lucide-react';

const TrainingHistoryAnalysis = () => {
  const [selectedMetric, setSelectedMetric] = useState('accuracy');
  
  // Dados do histórico de treinamento extraídos do arquivo
  const trainingData = useMemo(() => {
    const accuracy = [0.7142857142857143, 0.7333333333333333, 0.7619047619047619, 0.7666666666666667, 0.8095238095238095, 0.8380952380952381, 0.8666666666666667, 0.8857142857142857, 0.8809523809523809, 0.8952380952380953, 0.9047619047619048, 0.9047619047619048, 0.9095238095238095, 0.9095238095238095, 0.9333333333333333, 0.9380952380952381, 0.9428571428571428, 0.9571428571428572, 0.9619047619047619, 0.9571428571428572, 0.9666666666666667, 0.9714285714285714, 0.9714285714285714, 0.9714285714285714, 0.9619047619047619, 0.9761904761904762, 0.9761904761904762, 0.9809523809523809, 0.9809523809523809, 1.0, 1.0, 1.0, 0.9952380952380953, 1.0, 1.0, 1.0, 0.9952380952380953, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    
    const loss = [0.8928571428571429, 0.8857142857142857, 0.8666666666666667, 0.8571428571428571, 0.8285714285714286, 0.7952380952380952, 0.7476190476190476, 0.7142857142857143, 0.7095238095238095, 0.6952380952380952, 0.6857142857142857, 0.6809523809523809, 0.6714285714285714, 0.6666666666666666, 0.6238095238095238, 0.6095238095238095, 0.6047619047619047, 0.5857142857142857, 0.5809523809523809, 0.5857142857142857, 0.5761904761904762, 0.5714285714285714, 0.5714285714285714, 0.5714285714285714, 0.5809523809523809, 0.5666666666666667, 0.5666666666666667, 0.5619047619047619, 0.5619047619047619, 0.5428571428571428, 0.5333333333333333, 0.5333333333333333, 0.5380952380952381, 0.5333333333333333, 0.5333333333333333, 0.5333333333333333, 0.5380952380952381, 0.5333333333333333, 0.5333333333333333, 0.5333333333333333, 0.5333333333333333, 0.5333333333333333, 0.5333333333333333, 0.5333333333333333, 0.5333333333333333, 0.5333333333333333, 0.5333333333333333, 0.5333333333333333, 0.5333333333333333, 0.5333333333333333];
    
    const val_accuracy = [0.6363636363636364, 0.7272727272727273, 0.7727272727272727, 0.8181818181818182, 0.8636363636363636, 0.8636363636363636, 0.8636363636363636, 0.8636363636363636, 0.9090909090909091, 0.8636363636363636, 0.9090909090909091, 0.9545454545454546, 0.8636363636363636, 0.9090909090909091, 1.0, 0.9545454545454546, 0.9545454545454546, 1.0, 0.9545454545454546, 1.0, 0.9545454545454546, 0.9545454545454546, 1.0, 1.0, 1.0, 0.9545454545454546, 1.0, 0.9545454545454546, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    
    const val_loss = [0.8363636363636364, 0.8181818181818182, 0.8090909090909091, 0.7727272727272727, 0.7363636363636363, 0.7727272727272727, 0.7272727272727273, 0.7090909090909091, 0.6818181818181818, 0.6818181818181818, 0.6363636363636364, 0.6454545454545455, 0.7272727272727273, 0.6363636363636364, 0.6090909090909091, 0.6454545454545455, 0.5909090909090909, 0.5454545454545454, 0.5909090909090909, 0.5454545454545454, 0.5909090909090909, 0.6363636363636364, 0.5454545454545454, 0.5000000000000000, 0.4545454545454545, 0.5454545454545454, 0.4545454545454545, 0.6090909090909091, 0.3636363636363636, 0.3636363636363636, 0.3181818181818182, 0.3636363636363636, 0.3636363636363636, 0.3181818181818182, 0.3181818181818182, 0.2727272727272727, 0.4545454545454545, 0.2727272727272727, 0.2727272727272727, 0.2727272727272727, 0.2727272727272727, 0.2727272727272727, 0.2727272727272727, 0.2727272727272727, 0.2727272727272727, 0.2727272727272727, 0.2727272727272727, 0.2727272727272727, 0.2727272727272727, 0.2727272727272727];
    
    return accuracy.map((acc, index) => ({
      epoch: index + 1,
      accuracy: acc,
      loss: loss[index],
      val_accuracy: val_accuracy[index],
      val_loss: val_loss[index]
    }));
  }, []);

  // Análise estatística
  const analysisData = useMemo(() => {
    const finalAccuracy = trainingData[trainingData.length - 1].accuracy;
    const finalValAccuracy = trainingData[trainingData.length - 1].val_accuracy;
    const maxAccuracy = Math.max(...trainingData.map(d => d.accuracy));
    const maxValAccuracy = Math.max(...trainingData.map(d => d.val_accuracy));
    const minLoss = Math.min(...trainingData.map(d => d.loss));
    const minValLoss = Math.min(...trainingData.map(d => d.val_loss));
    
    return {
      finalAccuracy,
      finalValAccuracy,
      maxAccuracy,
      maxValAccuracy,
      minLoss,
      minValLoss,
      epochs: trainingData.length,
      convergence: trainingData.findIndex(d => d.accuracy >= 0.95) + 1
    };
  }, [trainingData]);

  const MetricCard = ({ title, value, icon: Icon, color = "blue", format = "percent" }) => (
    <div className={`bg-white p-6 rounded-lg shadow-lg border-l-4 border-${color}-500`}>
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <p className="text-2xl font-bold text-gray-900">
            {format === "percent" ? `${(value * 100).toFixed(1)}%` : value}
          </p>
        </div>
        <Icon className={`h-8 w-8 text-${color}-500`} />
      </div>
    </div>
  );

  const metrics = [
    { name: 'accuracy', label: 'Acurácia de Treino', color: '#3b82f6' },
    { name: 'val_accuracy', label: 'Acurácia de Validação', color: '#10b981' },
    { name: 'loss', label: 'Perda de Treino', color: '#ef4444' },
    { name: 'val_loss', label: 'Perda de Validação', color: '#f59e0b' }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Análise do Histórico de Treinamento
          </h1>
          <p className="text-lg text-gray-600">
            CNN Multiview para Classificação de Formas 3D
          </p>
        </div>

        {/* Métricas Principais */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <MetricCard
            title="Acurácia Final (Treino)"
            value={analysisData.finalAccuracy}
            icon={Target}
            color="green"
          />
          <MetricCard
            title="Acurácia Final (Validação)"
            value={analysisData.finalValAccuracy}
            icon={Award}
            color="blue"
          />
          <MetricCard
            title="Épocas Treinadas"
            value={analysisData.epochs}
            icon={Zap}
            color="purple"
            format="number"
          />
          <MetricCard
            title="Convergência (95%)"
            value={analysisData.convergence}
            icon={TrendingUp}
            color="orange"
            format="number"
          />
        </div>

        {/* Seletor de Métricas */}
        <div className="bg-white p-6 rounded-lg shadow-lg mb-8">
          <h3 className="text-lg font-semibold mb-4">Selecione as Métricas para Visualizar:</h3>
          <div className="flex flex-wrap gap-2">
            {metrics.map((metric) => (
              <button
                key={metric.name}
                onClick={() => setSelectedMetric(metric.name)}
                className={`px-4 py-2 rounded-md font-medium transition-colors ${
                  selectedMetric === metric.name
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                {metric.label}
              </button>
            ))}
          </div>
        </div>

        {/* Gráfico Principal */}
        <div className="bg-white p-6 rounded-lg shadow-lg mb-8">
          <h3 className="text-lg font-semibold mb-4">Evolução das Métricas de Treinamento</h3>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={trainingData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="epoch" />
              <YAxis />
              <Tooltip 
                formatter={(value, name) => [
                  `${(value * 100).toFixed(2)}%`,
                  name === 'accuracy' ? 'Acurácia (Treino)' :
                  name === 'val_accuracy' ? 'Acurácia (Validação)' :
                  name === 'loss' ? 'Perda (Treino)' :
                  'Perda (Validação)'
                ]}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="accuracy"
                stroke="#3b82f6"
                strokeWidth={2}
                name="Acurácia (Treino)"
                dot={{ fill: '#3b82f6', strokeWidth: 2, r: 3 }}
              />
              <Line
                type="monotone"
                dataKey="val_accuracy"
                stroke="#10b981"
                strokeWidth={2}
                name="Acurácia (Validação)"
                dot={{ fill: '#10b981', strokeWidth: 2, r: 3 }}
              />
              <Line
                type="monotone"
                dataKey="loss"
                stroke="#ef4444"
                strokeWidth={2}
                name="Perda (Treino)"
                dot={{ fill: '#ef4444', strokeWidth: 2, r: 3 }}
              />
              <Line
                type="monotone"
                dataKey="val_loss"
                stroke="#f59e0b"
                strokeWidth={2}
                name="Perda (Validação)"
                dot={{ fill: '#f59e0b', strokeWidth: 2, r: 3 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Análise Detalhada */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Resultados Finais */}
          <div className="bg-white p-6 rounded-lg shadow-lg">
            <h3 className="text-lg font-semibold mb-4 flex items-center">
              <Award className="h-5 w-5 mr-2 text-green-500" />
              Resultados Finais
            </h3>
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Acurácia de Treino:</span>
                <span className="font-bold text-green-600">
                  {(analysisData.finalAccuracy * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Acurácia de Validação:</span>
                <span className="font-bold text-blue-600">
                  {(analysisData.finalValAccuracy * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Perda Final (Treino):</span>
                <span className="font-bold text-red-600">
                  {analysisData.minLoss.toFixed(3)}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Perda Final (Validação):</span>
                <span className="font-bold text-orange-600">
                  {analysisData.minValLoss.toFixed(3)}
                </span>
              </div>
            </div>
          </div>

          {/* Insights */}
          <div className="bg-white p-6 rounded-lg shadow-lg">
            <h3 className="text-lg font-semibold mb-4 flex items-center">
              <TrendingUp className="h-5 w-5 mr-2 text-blue-500" />
              Insights do Treinamento
            </h3>
            <div className="space-y-3">
              <div className="flex items-start space-x-2">
                <div className="w-2 h-2 bg-green-500 rounded-full mt-2"></div>
                <p className="text-sm text-gray-700">
                  <strong>Excelente Performance:</strong> Modelo atingiu 100% de acurácia no conjunto de treino
                </p>
              </div>
              <div className="flex items-start space-x-2">
                <div className="w-2 h-2 bg-blue-500 rounded-full mt-2"></div>
                <p className="text-sm text-gray-700">
                  <strong>Generalização:</strong> Validação também atingiu 100%, indicando boa generalização
                </p>
              </div>
              <div className="flex items-start space-x-2">
                <div className="w-2 h-2 bg-purple-500 rounded-full mt-2"></div>
                <p className="text-sm text-gray-700">
                  <strong>Convergência Rápida:</strong> Atingiu 95% de acurácia na época {analysisData.convergence}
                </p>
              </div>
              <div className="flex items-start space-x-2">
                <div className="w-2 h-2 bg-orange-500 rounded-full mt-2"></div>
                <p className="text-sm text-gray-700">
                  <strong>Estabilidade:</strong> Métricas estabilizaram após época 30
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Comparação com Benchmarks */}
        <div className="bg-white p-6 rounded-lg shadow-lg mt-8">
          <h3 className="text-lg font-semibold mb-4">Comparação com Benchmarks</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <h4 className="font-semibold text-gray-700">CNN Tradicional</h4>
              <p className="text-2xl font-bold text-gray-500">85-90%</p>
              <p className="text-sm text-gray-600">Acurácia típica</p>
            </div>
            <div className="text-center p-4 bg-blue-50 rounded-lg">
              <h4 className="font-semibold text-blue-700">Multiview CNN</h4>
              <p className="text-2xl font-bold text-blue-600">95-98%</p>
              <p className="text-sm text-blue-600">Estado da arte</p>
            </div>
            <div className="text-center p-4 bg-green-50 rounded-lg">
              <h4 className="font-semibold text-green-700">Nosso Modelo</h4>
              <p className="text-2xl font-bold text-green-600">100%</p>
              <p className="text-sm text-green-600">Resultado obtido</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrainingHistoryAnalysis;