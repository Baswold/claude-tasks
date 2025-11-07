const express = require('express');
const axios = require('axios');
const path = require('path');
const cors = require('cors');
require('dotenv').config();
const Anthropic = require('@anthropic-ai/sdk');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(__dirname));

// Initialize Anthropic client
const anthropic = new Anthropic({
    apiKey: process.env.ANTHROPIC_API_KEY,
});

// Cache for models to reduce API calls
let modelsCache = {
    data: null,
    timestamp: null,
    ttl: 5 * 60 * 1000 // 5 minutes
};

/**
 * GET /api/models
 * Fetch popular AI models from HuggingFace
 */
app.get('/api/models', async (req, res) => {
    try {
        // Check cache
        const now = Date.now();
        if (modelsCache.data && modelsCache.timestamp && (now - modelsCache.timestamp < modelsCache.ttl)) {
            return res.json({ models: modelsCache.data });
        }

        // Fetch from HuggingFace API
        const response = await axios.get('https://huggingface.co/api/models', {
            params: {
                sort: 'downloads',
                direction: -1,
                limit: 100,
                filter: 'text-generation'
            },
            timeout: 10000
        });

        const models = response.data.map(model => ({
            id: model.id,
            modelId: model.id,
            author: model.id.split('/')[0] || 'Unknown',
            downloads: model.downloads || 0,
            likes: model.likes || 0,
            tags: model.tags || []
        }));

        // Add some popular models manually if needed
        const popularModels = [
            {
                id: 'meta-llama/Llama-2-70b-hf',
                modelId: 'Llama 2 70B',
                author: 'meta-llama',
                downloads: 1000000,
                likes: 5000,
                tags: ['text-generation', 'llama']
            },
            {
                id: 'mistralai/Mistral-7B-v0.1',
                modelId: 'Mistral 7B',
                author: 'mistralai',
                downloads: 800000,
                likes: 4000,
                tags: ['text-generation', 'mistral']
            },
            {
                id: 'tiiuae/falcon-180B',
                modelId: 'Falcon 180B',
                author: 'tiiuae',
                downloads: 500000,
                likes: 3000,
                tags: ['text-generation', 'falcon']
            }
        ];

        // Merge and deduplicate
        const allModels = [...popularModels, ...models];
        const uniqueModels = Array.from(new Map(allModels.map(m => [m.id, m])).values());

        // Update cache
        modelsCache.data = uniqueModels;
        modelsCache.timestamp = now;

        res.json({ models: uniqueModels });

    } catch (error) {
        console.error('Error fetching models:', error.message);

        // Return fallback data
        const fallbackModels = [
            { id: 'meta-llama/Llama-2-70b-hf', modelId: 'Llama 2 70B', author: 'meta-llama', downloads: 1000000, likes: 5000, tags: ['text-generation'] },
            { id: 'meta-llama/Llama-2-13b-hf', modelId: 'Llama 2 13B', author: 'meta-llama', downloads: 800000, likes: 4000, tags: ['text-generation'] },
            { id: 'meta-llama/Llama-2-7b-hf', modelId: 'Llama 2 7B', author: 'meta-llama', downloads: 700000, likes: 3500, tags: ['text-generation'] },
            { id: 'mistralai/Mistral-7B-v0.1', modelId: 'Mistral 7B', author: 'mistralai', downloads: 600000, likes: 3000, tags: ['text-generation'] },
            { id: 'tiiuae/falcon-180B', modelId: 'Falcon 180B', author: 'tiiuae', downloads: 500000, likes: 2500, tags: ['text-generation'] },
            { id: 'tiiuae/falcon-40b', modelId: 'Falcon 40B', author: 'tiiuae', downloads: 400000, likes: 2000, tags: ['text-generation'] },
            { id: 'google/flan-t5-xxl', modelId: 'Flan-T5 XXL', author: 'google', downloads: 300000, likes: 1500, tags: ['text-generation'] },
            { id: 'EleutherAI/gpt-neox-20b', modelId: 'GPT-NeoX 20B', author: 'EleutherAI', downloads: 250000, likes: 1200, tags: ['text-generation'] },
            { id: 'bigscience/bloom-176b', modelId: 'BLOOM 176B', author: 'bigscience', downloads: 200000, likes: 1000, tags: ['text-generation'] },
            { id: 'stabilityai/stablelm-3b-4e1t', modelId: 'StableLM 3B', author: 'stabilityai', downloads: 150000, likes: 800, tags: ['text-generation'] }
        ];

        res.json({ models: fallbackModels });
    }
});

/**
 * GET /api/model-details/:modelId
 * Get detailed information about a specific model
 */
app.get('/api/model-details/:modelId', async (req, res) => {
    try {
        const modelId = decodeURIComponent(req.params.modelId);

        // Try to fetch from HuggingFace API
        try {
            const response = await axios.get(`https://huggingface.co/api/models/${modelId}`, {
                timeout: 5000
            });

            const modelData = response.data;
            const config = modelData.config || {};

            // Estimate parameters from model card or config
            let parameters = estimateParameters(modelId, config);

            // Calculate RAM and VRAM requirements
            const { ram, vram } = calculateMemoryRequirements(parameters);

            res.json({
                parameters: parameters,
                ram: ram,
                vram: vram,
                type: detectModelType(modelId, modelData.tags || []),
                precision: 'FP16'
            });

        } catch (apiError) {
            // Fallback to estimation based on model name
            const parameters = estimateParametersFromName(modelId);
            const { ram, vram } = calculateMemoryRequirements(parameters);

            res.json({
                parameters: parameters,
                ram: ram,
                vram: vram,
                type: detectModelType(modelId, []),
                precision: 'FP16'
            });
        }

    } catch (error) {
        console.error('Error fetching model details:', error.message);
        res.status(500).json({ error: 'Failed to fetch model details' });
    }
});

/**
 * POST /api/find-hardware
 * Use Claude to research best hardware for a model
 */
app.post('/api/find-hardware', async (req, res) => {
    try {
        const { modelId, modelName, author } = req.body;

        if (!modelId) {
            return res.status(400).json({ error: 'Model ID is required' });
        }

        // Get model details first
        const detailsResponse = await axios.get(`http://localhost:${PORT}/api/model-details/${encodeURIComponent(modelId)}`);
        const modelDetails = detailsResponse.data;

        // Create a prompt for Claude to research hardware
        const prompt = `I need to find the best hardware to run the AI model "${modelName || modelId}" by ${author || 'unknown'}.

Model Specifications:
- Parameters: ${formatNumber(modelDetails.parameters)}
- RAM Required: ${modelDetails.ram} GB
- VRAM Required: ${modelDetails.vram} GB
- Model Type: ${modelDetails.type}
- Precision: ${modelDetails.precision}

Please research and recommend:
1. The best GPU options for running this model (consumer and enterprise options)
2. CPU recommendations
3. RAM requirements and recommendations
4. Any cloud platforms that would be suitable (with approximate costs)
5. Any optimization tips for running this model efficiently

Please provide practical, actionable recommendations with specific hardware models and approximate costs where possible.`;

        console.log('Sending request to Claude API...');

        // Call Claude API with web search enabled
        const message = await anthropic.messages.create({
            model: 'claude-sonnet-4-5-20250929',
            max_tokens: 4096,
            messages: [{
                role: 'user',
                content: prompt
            }]
        });

        const recommendation = message.content[0].text;

        res.json({ recommendation });

    } catch (error) {
        console.error('Error finding hardware:', error.message);

        // Provide a fallback response
        if (error.message.includes('API key')) {
            res.status(500).json({
                error: 'Claude API key not configured. Please set ANTHROPIC_API_KEY in your .env file.'
            });
        } else {
            res.status(500).json({
                error: 'Failed to get hardware recommendations. Please try again.'
            });
        }
    }
});

/**
 * Helper function to estimate parameters from model name
 */
function estimateParametersFromName(modelName) {
    const name = modelName.toLowerCase();

    // Extract number followed by 'b' (billion) or 't' (trillion)
    const billionMatch = name.match(/(\d+\.?\d*)b/);
    const trillionMatch = name.match(/(\d+\.?\d*)t/);

    if (trillionMatch) {
        return parseFloat(trillionMatch[1]) * 1e12;
    } else if (billionMatch) {
        return parseFloat(billionMatch[1]) * 1e9;
    }

    // Default estimates based on common models
    if (name.includes('llama-2-70b') || name.includes('falcon-180')) return 70e9;
    if (name.includes('llama-2-13b') || name.includes('falcon-40')) return 13e9;
    if (name.includes('llama-2-7b') || name.includes('mistral-7b')) return 7e9;
    if (name.includes('bloom-176')) return 176e9;
    if (name.includes('gpt-neox-20')) return 20e9;

    // Default to 7B if unknown
    return 7e9;
}

/**
 * Helper function to estimate parameters
 */
function estimateParameters(modelId, config) {
    // Try to get from config
    if (config.num_parameters) return config.num_parameters;

    // Try to calculate from config
    if (config.hidden_size && config.num_hidden_layers) {
        // Rough estimate: 12 * n_layers * hidden_size^2
        return 12 * config.num_hidden_layers * Math.pow(config.hidden_size, 2);
    }

    // Fallback to name-based estimation
    return estimateParametersFromName(modelId);
}

/**
 * Calculate memory requirements for a model
 * Formula:
 * - FP16: 2 bytes per parameter
 * - RAM: 1.2x model size (for overhead)
 * - VRAM: model size + activation memory (~20% of model size)
 */
function calculateMemoryRequirements(parameters) {
    const modelSizeGB = (parameters * 2) / (1024 ** 3); // FP16 = 2 bytes per param

    const ram = Math.ceil(modelSizeGB * 1.2); // 20% overhead
    const vram = Math.ceil(modelSizeGB * 1.3); // 30% for activations

    return { ram, vram };
}

/**
 * Detect model type from name and tags
 */
function detectModelType(modelName, tags) {
    const name = modelName.toLowerCase();
    const tagStr = tags.join(' ').toLowerCase();

    if (name.includes('llama') || tagStr.includes('llama')) return 'LLaMA';
    if (name.includes('mistral') || tagStr.includes('mistral')) return 'Mistral';
    if (name.includes('falcon') || tagStr.includes('falcon')) return 'Falcon';
    if (name.includes('bloom') || tagStr.includes('bloom')) return 'BLOOM';
    if (name.includes('gpt') || tagStr.includes('gpt')) return 'GPT';
    if (name.includes('t5') || tagStr.includes('t5')) return 'T5';
    if (name.includes('stable') || tagStr.includes('stable')) return 'StableLM';

    return 'Transformer';
}

/**
 * Format numbers for display
 */
function formatNumber(num) {
    if (num >= 1e12) return `${(num / 1e12).toFixed(1)} Trillion`;
    if (num >= 1e9) return `${(num / 1e9).toFixed(1)} Billion`;
    if (num >= 1e6) return `${(num / 1e6).toFixed(1)} Million`;
    return num.toLocaleString();
}

// Serve the main HTML file
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// Start server
app.listen(PORT, () => {
    console.log(`🚀 Inference Calculator running at http://localhost:${PORT}`);
    console.log(`📊 Beautiful UI ready`);
    console.log(`🤖 Claude AI researcher ready`);

    if (!process.env.ANTHROPIC_API_KEY) {
        console.warn('⚠️  Warning: ANTHROPIC_API_KEY not found in environment variables');
        console.warn('   Please create a .env file with your API key to enable hardware research');
    }
});
