import React, { useState } from 'react';
import './Brain_mri.css';

function Brain_mri() {
    const [image, setImage] = useState(null);
    const [prediction, setPrediction] = useState("");

    const handleImageChange = (e) => {
        setImage(e.target.files[0]);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        const formData = new FormData();
        formData.append('file', image);

        const response = await fetch('http://localhost:8000/predict_brain_mri/', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        setPrediction(result.prediction);
    };

    return (
        <div className="upload-container">
            <h2>Brain-MRI Classification</h2>
            <h3>['Glioma','Meningioma','No Tumor','Pituitary']</h3>
            <form onSubmit={handleSubmit} className="upload-form">
                <input type="file" onChange={handleImageChange} className="file-input" />
                <button type="submit" className="upload-button">Upload and Predict</button>
            </form>
            {prediction && <p className="prediction-result">Prediction: {prediction}</p>}
        </div>
    );
}

export default Brain_mri