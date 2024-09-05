import React from 'react'
import { Link } from 'react-router-dom';
import './Home.css';

function Home() {
  return (
    <div>
      <header className="title">Radiologix</header>
    <div className='home-cont'>
      <h1 className="home-title">Welcome to World of RadioLogy</h1>
      <h2 className='head'>Please choose the scan type</h2>
      <div className="role-links">
        <Link to="/lumbar" className="role-link">Lumbar</Link>
        <Link to="/brain_mri" className="role-link">Brain MRI</Link>
        <Link to="/pneumonia" className="role-link">Pneumonia</Link>
        <Link to="/covid_19" className="role-link">Covid 19</Link>
      </div>
    </div>
    </div>
  );
}

export default Home