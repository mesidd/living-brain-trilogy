'use client';

import { useState } from 'react';

export default function Home() {
  const [prompt, setPrompt] = useState('');
  const [response, setResponse] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setIsLoading(true);
    setResponse('');

    try {
      const res = await fetch('http://127.0.0.1:8000/api/prompt', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt }),
      });

      if (!res.ok) {
        throw new Error('Something went wrong');
      }

      const data = await res.json();
      setResponse(data.response);
    } catch (error) {
      console.error(error);
      setResponse('Failed to get a response from the server.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center bg-gray-900 text-white p-12">
      <h1 className="text-5xl font-bold mb-8">Living Brain Interface</h1>
      <p className="text-lg text-gray-400 mb-10">
        This is the genesis of your personal AI. Ask it anything.
      </p>

      <form onSubmit={handleSubmit} className="w-full max-w-2xl">
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Why is the sky blue?"
          className="w-full p-4 bg-gray-800 border border-gray-700 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none"
          rows={4}
        />
        <button
          type="submit"
          disabled={isLoading}
          className="w-full mt-4 px-4 py-3 bg-blue-600 rounded-lg font-semibold hover:bg-blue-700 disabled:bg-gray-500"
        >
          {isLoading ? 'Thinking...' : 'Ask'}
        </button>
      </form>

      {response && (
        <div className="w-full max-w-2xl mt-10 p-6 bg-gray-800 border border-gray-700 rounded-lg">
          <h2 className="text-2xl font-semibold mb-4">Response:</h2>
          <p className="text-gray-300 whitespace-pre-wrap">{response}</p>
        </div>
      )}
    </main>
  );
}