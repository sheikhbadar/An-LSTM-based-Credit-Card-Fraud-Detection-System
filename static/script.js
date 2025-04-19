document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('fraudForm');
    const resultDiv = document.getElementById('result');
    const loadingDiv = document.getElementById('loading');

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Show loading
        loadingDiv.style.display = 'block';
        resultDiv.style.display = 'none';
        
        // Get form data
        const formData = {
            user_id: 'test_user',  // In a real application, this would come from user authentication
            amount: document.getElementById('amount').value,
            time: document.getElementById('time').value,
            location: document.getElementById('location').value,
            device: document.getElementById('device').value,
            usual_time: document.getElementById('usual_time').value,
            usual_location: document.getElementById('usual_location').value,
            usual_device: document.getElementById('usual_device').value
        };
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Display results
            resultDiv.innerHTML = `
                <h3>Prediction Result:</h3>
                <p><strong>Status:</strong> ${data.prediction}</p>
                <p><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(2)}%</p>
                <h4>Risk Factors:</h4>
                <ul>
                    ${data.risk_factors.map(factor => `<li>${factor}</li>`).join('')}
                </ul>
            `;
            
            resultDiv.style.display = 'block';
        } catch (error) {
            resultDiv.innerHTML = `<p class="error">Error: ${error.message}</p>`;
            resultDiv.style.display = 'block';
        } finally {
            loadingDiv.style.display = 'none';
        }
    });
}); 