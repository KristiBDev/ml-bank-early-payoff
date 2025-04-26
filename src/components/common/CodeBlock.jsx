import React, { useState } from 'react';

const CodeBlock = ({ title, codeContent }) => {
  const [isVisible, setIsVisible] = useState(false);
  
  return (
    <div className="code-container">
      <div 
        className="code-header"
        onClick={() => setIsVisible(!isVisible)}
      >
        <span className="code-title">{title}</span>
      </div>
      {isVisible && (
        <pre className="code-content">
          <code>{codeContent}</code>
        </pre>
      )}
    </div>
  );
};

export default CodeBlock;
