
import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card'
import { Button } from './components/ui/button'
import { Input } from './components/ui/input'
import { Badge } from './components/ui/badge'
import { Progress } from './components/ui/progress'
import { motion } from 'framer-motion'
import { useQuery } from '@tanstack/react-query'
import { Newspaper, ShieldCheck, AlertCircle } from 'lucide-react'

interface Prediction {
  real_probability: number
  fake_probability: number
  verdict: string
  confidence: number
  word_count: number
}

const App: React.FC = () => {
  const [inputText, setInputText] = useState('')
  const [prediction, setPrediction] = useState<Prediction | null>(null)

  const { data, isLoading, error, refetch } = useQuery<Prediction>({
    queryKey: ['predict', inputText],
    queryFn: async () => {
      const res = await fetch('http://localhost:8001/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: inputText })
      })
      if (!res.ok) throw new Error('Prediction failed')
      return res.json()
    },
    enabled: false
  })

  const analyze = () => {
    if (inputText.trim()) {
      refetch()
    }
  }

  const getVerdictColor = (verdict: string) => {
    return verdict === 'REAL' ? 'bg-green-500' : 'bg-red-500'
  }

  const getIcon = (verdict: string) => {
    return verdict === 'REAL' ? ShieldCheck : AlertCircle
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      <div className="container mx-auto px-4 py-12 max-w-4xl">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <div className="inline-flex items-center gap-3 bg-white/80 backdrop-blur-xl rounded-2xl p-6 shadow-2xl border border-white/50">
            <Newspaper className="w-12 h-12 text-blue-600" />
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                Fake News Detector
              </h1>
              <p className="text-xl text-gray-600 mt-2">
                AI-Powered News Verification (99.8% Accuracy)
              </p>
            </div>
          </div>
        </motion.div>

        <div className="grid md:grid-cols-2 gap-8 mb-8">
          <Card>
            <CardHeader>
              <CardTitle>Analyze News</CardTitle>
              <CardDescription>
                Paste title + full article text. Headlines work with confidence threshold.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Paste news title + full text here..."
                className="w-full h-32 p-4 border border-gray-300 rounded-lg resize-vertical focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              <Button 
                onClick={analyze}
                disabled={!inputText.trim() || isLoading}
                className="w-full mt-4 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700"
              >
                {isLoading ? 'Analyzing...' : '🔍 Analyze News'}
              </Button>
            </CardContent>
          </Card>

          {prediction && (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="space-y-4"
            >
              <Card>
                <CardHeader className="flex flex-row items-center gap-3">
                  <div className={`p-3 rounded-xl ${getVerdictColor(prediction.verdict)} shadow-lg shadow-${prediction.verdict === 'REAL' ? 'green' : 'red'}/20`}>
                    {React.createElement(getIcon(prediction.verdict), { className: 'w-8 h-8 text-white' })}
                  </div>
                  <div>
                    <CardTitle className="text-2xl font-bold capitalize">
                      {prediction.verdict === 'REAL' ? '✅ Real News' : '❌ Fake News'}
                    </CardTitle>
                    <p className="text-3xl font-black text-gray-900">
                      {prediction.confidence * 100}%
                    </p>
                    <p className="text-sm text-gray-500">Confidence Score</p>
                  </div>
                </CardHeader>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Confidence Breakdown</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Real News</span>
                      <span>{(prediction.real_probability * 100).toFixed(1)}%</span>
                    </div>
                    <Progress value={prediction.real_probability * 100} className="h-3" />
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Fake News</span>
                      <span>{(prediction.fake_probability * 100).toFixed(1)}%</span>
                    </div>
                    <Progress value={prediction.fake_probability * 100} className="h-3" />
                  </div>
                  <div className="grid grid-cols-3 gap-2 text-sm">
                    <Badge variant="secondary">{prediction.word_count} words</Badge>
                    <Badge variant="outline">Threshold 0.3</Badge>
                    <Badge>XGBoost 99.8%</Badge>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          )}
        </div>

        {error && (
          <Card className="bg-red-50 border-red-200">
            <CardContent className="p-6">
              <div className="flex items-center gap-3 text-red-800">
                <AlertCircle className="w-6 h-6" />
                <p>Backend unavailable. Ensure API running on localhost:8001</p>
              </div>
            </CardContent>
          </Card>
        )}

        <div className="mt-16 pt-12 border-t border-gray-200">
          <div className="text-center space-y-2 text-sm text-gray-500">
            <p>Powered by XGBoost ML • 99.8% Test Accuracy</p>
            <p>Backend: FastAPI • Frontend: React + Tailwind</p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App

