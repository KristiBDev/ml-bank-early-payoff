import React, { useState, useEffect } from 'react';
import './styles.css';

// Import the section components with correct paths
import Introduction from './components/sections/Introduction';
import DataPreparation from './components/sections/DataPreparation';
import ExploratoryAnalysis from './components/sections/ExploratoryAnalysis';
import FeatureEngineering from './components/sections/FeatureEngineering';
import ModelTraining from './components/sections/ModelTraining';
import Evaluation from './components/sections/Evaluation';
import ExportResults from './components/sections/ExportResults';

// Import common components
import CodeBlock from './components/common/CodeBlock';

const Home = () => {
  const [activeSection, setActiveSection] = useState('introduction');
  
  // Function to handle smooth scrolling to sections
  const scrollToSection = (sectionId) => {
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
      setActiveSection(sectionId);
    }
  };
  
  // Fix the scroll handler to ensure sidebar follows when scrolling
  useEffect(() => {
    // Only add this effect in environments with window object (browser)
    if (typeof window !== 'undefined') {
      const handleScroll = () => {
        // Simple scroll handler just to update active section
        const sections = document.querySelectorAll('.section');
        const scrollPosition = window.scrollY + 100;
        
        sections.forEach(section => {
          if (section.offsetTop <= scrollPosition && 
              section.offsetTop + section.offsetHeight > scrollPosition) {
            setActiveSection(section.id);
          }
        });
      };
      
      window.addEventListener('scroll', handleScroll);
      return () => window.removeEventListener('scroll', handleScroll);
    }
  }, []);

  return (
    <div className="case-study-container">
      <header className="header">
        <div className="header-content">
          <h1>Banking Case Study</h1>
          <p>Data Analysis and Machine Learning Solution for Early Loan Payoff Classification</p>
        </div>
      </header>
      
      <div className="main-content">
        <nav className="sidebar">
          <ul>
            <li className={activeSection === 'introduction' ? 'active' : ''}>
              <a onClick={() => scrollToSection('introduction')}>Introduction</a>
            </li>
            <li className={activeSection === 'data-analysis' ? 'active' : ''}>
              <a onClick={() => scrollToSection('data-analysis')}>Data Preparation</a>
            </li>
            <li className={activeSection === 'feature-engineering' ? 'active' : ''}>
              <a onClick={() => scrollToSection('feature-engineering')}>Exploratory Analysis</a>
            </li>
            <li className={activeSection === 'model-training' ? 'active' : ''}>
              <a onClick={() => scrollToSection('model-training')}>Feature Engineering</a>
            </li>
            <li className={activeSection === 'evaluation' ? 'active' : ''}>
              <a onClick={() => scrollToSection('evaluation')}>Modeling</a>
            </li>
            <li className={activeSection === 'export-results' ? 'active' : ''}>
              <a onClick={() => scrollToSection('export-results')}>Evaluation</a>
            </li>
            <li className={activeSection === 'conclusion' ? 'active' : ''}>
              <a onClick={() => scrollToSection('conclusion')}>Export Results</a>
            </li>
          </ul>
        </nav>
        
        <div className="content">
          {/* Section components will be placed here */}
          <section id="introduction" className="section">
            <Introduction />
          </section>
          
          <section id="data-analysis" className="section">
            <DataPreparation />
          </section>
          
          <section id="feature-engineering" className="section">
            <ExploratoryAnalysis /> 
          </section>
          
          <section id="model-training" className="section">
            <FeatureEngineering />
          </section>
          
          <section id="evaluation" className="section">
            <ModelTraining />
          </section>
          
          <section id="export-results" className="section">
            <Evaluation />
          </section>
          
          <section id="conclusion" className="section">
            <ExportResults />
          </section>
        </div>
      </div>
      
      <footer className="footer">
        <div className="footer-content">
          <p> 2025 Banking Case Study Machine Learning Project</p>
          <div className="footer-links">
            
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Home;