'use client';

import { useState } from 'react';

// Define a type for our synthesis result for cleaner state management
type SynthesisResult = {
  thesis: string;
  antithesis: string;
  conflict: string;
  synthesis: string;
};

export default function HegelEnginePage() {
  const [topic, setTopic] = useState('social media');
  const [result, setResult] = useState<SynthesisResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setIsLoading(true);
    setResult(null);

    try {
      const res = await fetch('http://127.0.0.1:8000/api/synthesize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic }),
      });

      if (!res.ok) throw new Error('Failed to get synthesis from the server.');
      
      const data = await res.json();
      setResult(data);
    } catch (error) {
      console.error(error);
      // A simple error state
      setResult({ thesis: '', antithesis: '', conflict: 'Error fetching synthesis.', synthesis: '' });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center bg-gray-900 text-white p-12">
      <h1 className="text-5xl font-bold mb-4">The Hegelian Engine</h1>
      <p className="text-lg text-gray-400 mb-10">
        An AI for Dialectical Reasoning: Thesis + Antithesis → Synthesis
      </p>

      <form onSubmit={handleSubmit} className="w-full max-w-lg mb-12">
        <input
          type="text"
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
          placeholder="Enter a topic..."
          className="w-full p-3 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-purple-500 focus:outline-none"
        />
        <button
          type="submit"
          disabled={isLoading}
          className="w-full mt-4 px-4 py-3 bg-purple-600 rounded-lg font-semibold hover:bg-purple-700 disabled:bg-gray-500"
        >
          {isLoading ? 'Synthesizing...' : 'Generate Synthesis'}
        </button>
      </form>

      {result && (
        <div className="w-full max-w-5xl">
          {/* Conflict Section */}
          <div className="mb-8 p-6 bg-gray-800 border-2 border-yellow-500 rounded-lg">
            <h2 className="text-2xl font-bold text-yellow-400 mb-3">Core Conflict</h2>
            <p className="text-gray-300 whitespace-pre-wrap">{result.conflict}</p>
          </div>

          {/* Synthesis Section */}
          <div className="p-6 bg-gradient-to-r from-purple-800 to-indigo-800 border-2 border-purple-500 rounded-lg">
            <h2 className="text-3xl font-bold text-white mb-4">Synthesis</h2>
            <p className="text-gray-200 whitespace-pre-wrap text-lg">{result.synthesis}</p>
          </div>
        </div>
      )}
    </main>
  );
}