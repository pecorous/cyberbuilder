from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import json
import os
from datetime import datetime
import logging
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
# IMPORTANT: Set your ChatGPT API key here or as an environment variable
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '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')  # Replace with your actual API key

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# System prompt for ChatGPT
SYSTEM_PROMPT = """You are a PC builder assistant. Given components and usage type, recommend a build.

Performance expectations by usage:
- Gaming: 60+ FPS at 1080p/1440p, handles AAA titles
- Workstation: Fast compile times, smooth multitasking, handles large projects  
- Content Creation: Adobe suite, 4K video editing, fast rendering
- AI/ML: CUDA acceleration, handles PyTorch/TensorFlow workloads
- Streaming: 1080p 60fps streaming while gaming
- General: Office, web browsing, light tasks

Return ONLY a JSON object:
{
    "components": [
        {"type": "CPU", "name": "exact name from data", "price": 99.99},
        {"type": "GPU", "name": "exact name from data", "price": 299.99},
        {"type": "Motherboard", "name": "exact name", "price": 99.99},
        {"type": "Memory", "name": "exact name", "price": 49.99},
        {"type": "Storage", "name": "exact name", "price": 59.99},
        {"type": "PSU", "name": "exact name", "price": 69.99},
        {"type": "Case", "name": "exact name", "price": 49.99},
        {"type": "CPU Cooler", "name": "exact name", "price": 29.99}
    ],
    "summary": "2-3 sentences about performance expectations."
}

Use ONLY items from provided data. Stay within budget."""

def load_latest_component_data(budget=1500):
    """Load component data with smart budget-based filtering"""
    try:
        logger.info(f"Fetching data for ${budget} budget...")
        
        # Define budget allocations and ranges
        budget_ranges = {
            'cpu': {'min': budget * 0.10, 'max': budget * 0.25, 'ideal': budget * 0.15},
            'gpu': {'min': budget * 0.20, 'max': budget * 0.40, 'ideal': budget * 0.30},
            'motherboard': {'min': budget * 0.08, 'max': budget * 0.15, 'ideal': budget * 0.10},
            'memory': {'min': budget * 0.05, 'max': budget * 0.12, 'ideal': budget * 0.08},
            'storage': {'min': budget * 0.05, 'max': budget * 0.10, 'ideal': budget * 0.07},
            'psu': {'min': budget * 0.05, 'max': budget * 0.10, 'ideal': budget * 0.07},
            'case': {'min': budget * 0.04, 'max': budget * 0.08, 'ideal': budget * 0.06},
            'cpu-cooler': {'min': budget * 0.02, 'max': budget * 0.08, 'ideal': budget * 0.04}
        }
        
        # Skip GPU for budget builds under $800
        if budget < 800:
            budget_ranges.pop('gpu', None)
        
        processed_data = {}
        
        # Component files to fetch
        file_map = {
            'cpu': 'cpu.json',
            'gpu': 'video-card.json',
            'motherboard': 'motherboard.json',
            'memory': 'memory.json',
            'storage': 'internal-hard-drive.json',
            'psu': 'power-supply.json',
            'case': 'case.json',
            'cpu-cooler': 'cpu-cooler.json'
        }
        
        for category, filename in file_map.items():
            if category not in budget_ranges:
                continue
                
            try:
                # Get raw file URL
                raw_url = f"https://raw.githubusercontent.com/docyx/pc-part-dataset/main/data/json/{filename}"
                response = requests.get(raw_url)
                response.raise_for_status()
                
                # Parse JSON
                items = response.json()
                
                # Get price range for this category
                price_range = budget_ranges[category]
                
                # Filter items within budget range
                filtered_items = []
                for item in items:
                    price = item.get('price')
                    if price and price > 0 and price_range['min'] <= price <= price_range['max']:
                        # Create minimal item (name and price only)
                        filtered_items.append({
                            'n': item.get('name', 'Unknown')[:50],  # Truncate long names
                            'p': round(float(price), 2)
                        })
                
                if filtered_items:
                    # Sort by price
                    filtered_items.sort(key=lambda x: x['p'])
                    
                    # Further limit items - keep only best options
                    if len(filtered_items) > 10:
                        # Keep items close to ideal price
                        ideal_price = price_range['ideal']
                        filtered_items.sort(key=lambda x: abs(x['p'] - ideal_price))
                        filtered_items = filtered_items[:10]
                    
                    processed_data[category] = filtered_items
                    logger.info(f"{category}: {len(filtered_items)} items (${price_range['min']:.0f}-${price_range['max']:.0f})")
                    
            except Exception as e:
                logger.error(f"Failed to load {category}: {e}")
        
        total_items = sum(len(items) for items in processed_data.values())
        logger.info(f"Total items loaded: {total_items}")
        return processed_data
        
    except Exception as e:
        logger.error(f"Failed to load component data: {e}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/api/generate-build', methods=['POST'])
def generate_build():
    """Generate PC build recommendations using ChatGPT"""
    try:
        # Get request data
        data = request.json
        budget = data.get('budget')
        usage_type = data.get('usageType')
        
        # Validate inputs
        if not budget or not usage_type:
            return jsonify({"error": "Budget and usage type are required"}), 400
        
        # Load the latest component data with budget filtering
        component_data = load_latest_component_data(budget)
        if not component_data:
            return jsonify({"error": "No component data available"}), 500
        
        # Prepare minimal prompt
        user_prompt = f"Budget: ${budget}, Usage: {usage_type}\n\nComponents (format: n=name, p=price):\n{json.dumps(component_data)}\n\nBuild a PC."

        # Call ChatGPT API
        logger.info(f"Generating build for budget: ${budget}, usage: {usage_type}")
        
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
                max_tokens=800
            )
            
            # Extract the response content
            ai_response = response.choices[0].message.content
            
        except Exception as e:
            logger.warning(f"API call failed: {e}")
            # Fallback
            return jsonify({"error": "Failed to generate build. Please try again."}), 503

        # Parse the response
        logger.info("Received response from ChatGPT")
        
        # Try to parse as JSON
        try:
            build_data = json.loads(ai_response)
        except json.JSONDecodeError:
            # If not valid JSON, try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                build_data = json.loads(json_match.group())
            else:
                # Fallback: create a basic response
                build_data = generate_fallback_build(budget, usage_type, component_data)
        
        # Expand abbreviated names back for display
        if 'components' in build_data:
            for component in build_data['components']:
                # Components have abbreviated keys, expand them
                if 'n' in component:
                    component['name'] = component.pop('n')
                if 'p' in component:
                    component['price'] = component.pop('p')
        
        # Add metadata
        build_data['metadata'] = {
            'generated_at': datetime.now().isoformat(),
            'budget': budget,
            'usage_type': usage_type,
            'ai_model': 'gpt-3.5-turbo'
        }
        
        return jsonify(build_data)
        
    except Exception as e:
        logger.error(f"Error in generate_build: {str(e)}")
        return jsonify({"error": "AI service error. Please try again later."}), 503

def generate_fallback_build(budget, usage_type, component_data):
    """Generate a fallback build if ChatGPT fails"""
    logger.info("Generating fallback build")
    
    # Basic allocation percentages based on usage type
    allocations = {
        'gaming': {'cpu': 0.15, 'gpu': 0.35, 'motherboard': 0.10, 'memory': 0.08, 'storage': 0.08, 'psu': 0.08, 'case': 0.06, 'cpu-cooler': 0.05},
        'workstation': {'cpu': 0.25, 'gpu': 0.20, 'motherboard': 0.12, 'memory': 0.15, 'storage': 0.10, 'psu': 0.08, 'case': 0.05, 'cpu-cooler': 0.05},
        'content-creation': {'cpu': 0.20, 'gpu': 0.25, 'motherboard': 0.10, 'memory': 0.12, 'storage': 0.12, 'psu': 0.08, 'case': 0.06, 'cpu-cooler': 0.05},
        'general': {'cpu': 0.20, 'gpu': 0.15, 'motherboard': 0.12, 'memory': 0.10, 'storage': 0.10, 'psu': 0.08, 'case': 0.08, 'cpu-cooler': 0.05},
        'streaming': {'cpu': 0.20, 'gpu': 0.30, 'motherboard': 0.10, 'memory': 0.10, 'storage': 0.08, 'psu': 0.08, 'case': 0.06, 'cpu-cooler': 0.05},
        'ai-ml': {'cpu': 0.15, 'gpu': 0.40, 'motherboard': 0.10, 'memory': 0.15, 'storage': 0.08, 'psu': 0.08, 'case': 0.04, 'cpu-cooler': 0.05}
    }
    
    allocation = allocations.get(usage_type, allocations['general'])
    components = []
    
    # Select components based on budget allocation
    for component_type, percentage in allocation.items():
        target_price = budget * percentage
        
        # Get components from data
        available = component_data.get(component_type, [])
        if not available:
            continue
            
        # Find best match for target price
        best_match = None
        best_diff = float('inf')
        
        for item in available:
            price = item.get('p', 0)
            if price:
                diff = abs(price - target_price)
                if diff < best_diff and price <= target_price * 1.2:  # Allow 20% over target
                    best_match = item
                    best_diff = diff
        
        if best_match:
            components.append({
                'type': component_type.upper().replace('-', ' '),
                'name': best_match.get('n', 'Unknown'),
                'price': best_match.get('p', 0)
            })
    
    # Generate notes based on usage type
    notes_map = {
        'gaming': "This build prioritizes GPU performance for smooth gaming at high settings. The selected components provide excellent price-to-performance ratio for modern games.",
        'workstation': "Built for productivity with emphasis on CPU performance and memory capacity. This configuration handles demanding professional workloads efficiently.",
        'content-creation': "Balanced build optimized for creative workflows with strong CPU and GPU performance. Perfect for video editing, 3D rendering, and content production.",
        'general': "Well-rounded system for everyday computing needs. Provides reliable performance for office work, web browsing, and light multimedia tasks.",
        'streaming': "Optimized for live streaming with balanced CPU and GPU performance. Handles simultaneous gaming and encoding without bottlenecks.",
        'ai-ml': "GPU-focused build designed for machine learning and AI workloads. Maximizes CUDA cores and VRAM for training models and data processing."
    }
    
    return {
        'components': components,
        'summary': notes_map.get(usage_type, notes_map['general'])
    }

@app.route('/api/fetch-latest-data', methods=['GET'])
def fetch_latest_data():
    """Endpoint to manually refresh data from GitHub"""
    try:
        data = load_latest_component_data()
        if data:
            total_parts = sum(len(items) for items in data.values())
            return jsonify({
                "status": "success",
                "message": f"Fetched {total_parts} components from GitHub",
                "categories": list(data.keys())
            })
        else:
            return jsonify({"error": "Failed to fetch data"}), 500
            
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/test-components', methods=['GET'])
def test_components():
    """Test endpoint that returns sample component data"""
    sample_data = {
        "cpu": [
            {"name": "AMD Ryzen 5 5600X", "price": 199, "specs": {"cores": "6", "speed": "3.7GHz"}},
            {"name": "Intel Core i5-12400F", "price": 149, "specs": {"cores": "6", "speed": "2.5GHz"}},
            {"name": "AMD Ryzen 7 5700X", "price": 249, "specs": {"cores": "8", "speed": "3.4GHz"}},
            {"name": "Intel Core i7-12700K", "price": 349, "specs": {"cores": "12", "speed": "3.6GHz"}}
        ],
        "gpu": [
            {"name": "NVIDIA RTX 4060", "price": 299, "specs": {"memory": "8GB"}},
            {"name": "AMD RX 6600", "price": 249, "specs": {"memory": "8GB"}},
            {"name": "NVIDIA RTX 4070", "price": 599, "specs": {"memory": "12GB"}},
            {"name": "AMD RX 7800 XT", "price": 499, "specs": {"memory": "16GB"}}
        ],
        "motherboard": [
            {"name": "MSI B550-A PRO", "price": 139, "specs": {"socket": "AM4", "form": "ATX"}},
            {"name": "ASUS TUF Gaming B660M", "price": 149, "specs": {"socket": "LGA1700", "form": "mATX"}},
            {"name": "Gigabyte B550 AORUS Elite", "price": 179, "specs": {"socket": "AM4", "form": "ATX"}}
        ],
        "memory": [
            {"name": "Corsair Vengeance LPX 16GB", "price": 45, "specs": {"speed": "3200MHz", "type": "DDR4"}},
            {"name": "G.Skill Ripjaws V 32GB", "price": 89, "specs": {"speed": "3600MHz", "type": "DDR4"}},
            {"name": "Kingston Fury Beast 16GB", "price": 55, "specs": {"speed": "3200MHz", "type": "DDR4"}}
        ],
        "storage": [
            {"name": "Samsung 970 EVO Plus 1TB", "price": 79, "specs": {"capacity": "1TB", "type": "NVMe"}},
            {"name": "WD Black SN770 1TB", "price": 69, "specs": {"capacity": "1TB", "type": "NVMe"}},
            {"name": "Crucial MX500 2TB", "price": 119, "specs": {"capacity": "2TB", "type": "SATA"}}
        ],
        "psu": [
            {"name": "Corsair RM650x", "price": 89, "specs": {"wattage": "650W", "efficiency": "80+ Gold"}},
            {"name": "EVGA SuperNOVA 750", "price": 99, "specs": {"wattage": "750W", "efficiency": "80+ Gold"}},
            {"name": "Seasonic Focus GX-550", "price": 79, "specs": {"wattage": "550W", "efficiency": "80+ Gold"}}
        ],
        "case": [
            {"name": "NZXT H510", "price": 69, "specs": {"form": "ATX", "color": "Black"}},
            {"name": "Corsair 4000D Airflow", "price": 79, "specs": {"form": "ATX", "color": "Black"}},
            {"name": "Lian Li Lancool 215", "price": 89, "specs": {"form": "ATX", "color": "Black"}}
        ],
        "cpu-cooler": [
            {"name": "Cooler Master Hyper 212", "price": 35, "specs": {"type": "Air", "tdp": "150W"}},
            {"name": "Arctic Freezer 34", "price": 32, "specs": {"type": "Air", "tdp": "150W"}},
            {"name": "be quiet! Dark Rock 4", "price": 69, "specs": {"type": "Air", "tdp": "200W"}}
        ]
    }
    return jsonify(sample_data)

if __name__ == '__main__':
    # Remove or comment out app.run()
    # app.run(debug=True, host='0.0.0.0', port=5000)
    pass

"""
BACKEND SETUP INSTRUCTIONS:

1. Install required packages:
   pip install flask flask-cors openai requests

2. Set your OpenAI API key:
   - Option 1: Set as environment variable
     export OPENAI_API_KEY="your-api-key-here"
   
   - Option 2: Replace 'YOUR_OPENAI_API_KEY_HERE' in the code (line 19)

3. Run the backend server:
   python backend.py

4. The server will run on http://localhost:5000

API ENDPOINTS:

1. POST /api/generate-build
   - Generates PC build recommendations
   - Body: {
       "budget": 1500,
       "usageType": "gaming"
     }

2. GET /api/test-components
   - Returns sample component data for testing

3. GET /health
   - Health check endpoint

TOKEN OPTIMIZATION:

- Only loads components within budget range
- Minimal data format (name + price only)
- ~80 components instead of 10,000+
- Reduces token usage by 99%

BUDGET LOGIC:

- CPU: 10-25% of budget
- GPU: 20-40% of budget (skipped under $800)
- Memory: 5-12% of budget
- Storage: 5-10% of budget
- PSU: 5-10% of budget
- Case: 4-8% of budget
- CPU Cooler: 2-8% of budget
"""
