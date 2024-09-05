import React from 'react';
import './App.css';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Home from './Components/Home';
import Brain_mri from './Components/Brain_mri';
import Covid_19 from './Components/Covid_19';
import Pneumonia from './Components/Pneumonia';
import Lumbar from './Components/Lumbar';

function App() {
  return (
    <div className="background-container">
    <Router>
      <Routes >
        <Route path='/' element={<Home/>} />
        <Route path='/brain_mri' element={<Brain_mri />} />
        <Route path='/covid_19' element={<Covid_19 />} />
        <Route path='/pneumonia' element={<Pneumonia />} />
        <Route path='/lumbar' element={<Lumbar />} />
      </Routes>
    </Router>
    </div>
  );
}

export default App;
