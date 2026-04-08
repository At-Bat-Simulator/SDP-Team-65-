import React from 'react';
import { Link } from 'react-router-dom';

const ModelDocs = () => {
  return (
    <div style={{ padding: '20px', fontFamily: 'Times New Roman' }}>
      <h1>Model Documentation</h1>
      <p>This page provides an overview of the machine learning models used in the At-Bat Simulator.</p>

      <h2>Simulation Accuracy</h2>
      <p>Comparison of our model's simulated outcomes against actual MLB averages.</p>
      
<table style={{ 
        width: '100%', 
        borderCollapse: 'collapse', 
        marginBottom: '30px',
        backgroundColor: '#2a2a2a', /* Dark background */
        color: '#ffffff',           /* Force text to be white */
        boxShadow: '0 4px 6px rgba(0,0,0,0.3)'
      }}>
        <thead>
          <tr style={{ backgroundColor: '#1a1a1a', textAlign: 'left' }}>
            <th style={{ padding: '12px', borderBottom: '2px solid #555' }}>Statistic</th>
            <th style={{ padding: '12px', borderBottom: '2px solid #555' }}>MLB Average (Statcast)</th>
            <th style={{ padding: '12px', borderBottom: '2px solid #555' }}>Our Model (200 At-Bats Simulated)</th>
            <th style={{ padding: '12px', borderBottom: '2px solid #555' }}>Our Model (500 At-Bats Simulated)</th>         
          </tr>
        </thead>
        <tbody>
          <tr>
            <td style={{ padding: '12px', borderBottom: '1px solid #444' }}>Swing %</td>
            <td style={{ padding: '12px', borderBottom: '1px solid #444' }}>~47%</td>
            <td style={{ padding: '12px', borderBottom: '1px solid #444' }}>TBD</td>
            <td style={{ padding: '12px', borderBottom: '1px solid #444' }}>TBD</td>
          </tr>
          <tr>
            <td style={{ padding: '12px', borderBottom: '1px solid #444' }}>Walk %</td>
            <td style={{ padding: '12px', borderBottom: '1px solid #444' }}>~9%</td>
            <td style={{ padding: '12px', borderBottom: '1px solid #444' }}>TBD</td>
            <td style={{ padding: '12px', borderBottom: '1px solid #444' }}>TBD</td>
          </tr>
          <tr>
            <td style={{ padding: '12px', borderBottom: '1px solid #444' }}>Strikeout % (K%)</td>
            <td style={{ padding: '12px', borderBottom: '1px solid #444' }}>~23%</td>
            <td style={{ padding: '12px', borderBottom: '1px solid #444' }}>TBD</td>
            <td style={{ padding: '12px', borderBottom: '1px solid #444' }}>TBD</td>
          </tr>
          <tr>
            <td style={{ padding: '12px', borderBottom: '1px solid #444' }}>Ball-in-Play %</td>
            <td style={{ padding: '12px', borderBottom: '1px solid #444' }}>~70%</td>
            <td style={{ padding: '12px', borderBottom: '1px solid #444' }}>TBD</td>
            <td style={{ padding: '12px', borderBottom: '1px solid #444' }}>TBD</td>
          </tr>
          <tr>
            <td style={{ padding: '12px', borderBottom: '1px solid #444' }}>Average Pitches</td>
            <td style={{ padding: '12px', borderBottom: '1px solid #444' }}>~3.9</td>
            <td style={{ padding: '12px', borderBottom: '1px solid #444' }}>TBD</td>
            <td style={{ padding: '12px', borderBottom: '1px solid #444' }}>TBD</td>
          </tr>
          <tr>
            <td style={{ padding: '12px', borderBottom: '1px solid #444' }}>First-Pitch Strike %</td>
            <td style={{ padding: '12px', borderBottom: '1px solid #444' }}>~60%</td>
            <td style={{ padding: '12px', borderBottom: '1px solid #444' }}>TBD</td>
            <td style={{ padding: '12px', borderBottom: '1px solid #444' }}>TBD</td>
          </tr>
        </tbody>
      </table>
      
      <h3>Pitch Type Model</h3>
      <p>Type: Long Short-Term Memory (LSTM) </p>
      <p>Summary: Predicting pitch type in baseball is inherently noisy, even MLB scouts with full game film and advance reports don't achieve much higher accuracy. The model performs well on high-frequency pitch types (FF, SI, FC) and shows lower performance on rarer or more situational pitches (CU, ST) due to class imbalance in the data.
      </p>
      <a href="https://github.com/At-Bat-Simulator/SDP-Team-65-/blob/main/Pitch%20Type%20Prediction/README.md" target="_blank" rel="noreferrer">
        View on GitHub
      </a>
      <h3>Pitch Location Model</h3>
      <p>Type: Gaussian Mixture Model (GMM) on top of LSTM</p>
      <p>Summary: The LSTM predicts a mean location that tends to regress toward the center of the strike zone — a known limitation of sequence-to-sequence regression models. Real pitchers don't throw everything down the middle; they work edges, bury breaking balls, and climb the ladder with fastballs. To capture the true shape of a pitcher's location distribution, we fit a Gaussian Mixture Model (GMM) separately on the full 2021–2024 Statcast dataset, grouped by pitch type.
</p>
      <a href="https://github.com/At-Bat-Simulator/SDP-Team-65-/blob/main/Pitch%20Location%20Prediction/README.md" target="_blank" rel="noreferrer">
        View on GitHub
      </a>

      <h3>Swing/Take Model</h3>
      <p>Type: Long Short-Term Memory (LSTM) </p>
      <p>Summary: The swing and take model utilizes the results from our Pitch type and location models to predict whether or not a batter will swing or take on a given pitch. The model was trained on the actual statcast dataset before being tested on the output dataset from our models for improved accuracy.</p>
      <a href="https://github.com/At-Bat-Simulator/SDP-Team-65-/tree/main/Offensive%20Models/swingtake" target="_blank" rel="noreferrer">
        View on GitHub
      </a>

      <h3>Launch Angle/Exit Velocity Model</h3>
      <p>Type: Long Short-Term Memory (LSTM) </p>
      <p>Summary: This model uses a LSTM to predict the exit velocity and launch angle of a hit. It uses the location of the pitch, the type of pitch, the swing angle and other features to calculate where, how fast and how far the baseball would go after making contact with a bat.</p>
      <a href="https://github.com/At-Bat-Simulator/SDP-Team-65-/tree/main/Offensive%20Models/EV_and_LA" target="_blank" rel="noreferrer">
        View on GitHub
      </a>

      <br /><br />
      <Link to="/">
        <button 
        type="button" >
          ← Back to Simulator
        </button>
      </Link>
    </div>
  );
};

export default ModelDocs;