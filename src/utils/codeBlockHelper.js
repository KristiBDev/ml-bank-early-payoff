/**
 * Creates a code block with a clickable header for toggling code visibility
 * @param {string} title - The title of the code block
 * @param {string} codeContent - The code content to display
 * @param {string} sectionKey - The key used in the visibleSections state
 * @param {boolean} isVisible - Whether the code section is currently visible
 * @param {Function} toggleFunction - The function to toggle visibility
 * @returns {JSX.Element} - The code block component
 */
export const createCodeBlock = (title, codeContent, sectionKey, isVisible, toggleFunction) => {
  return (
    <div className="code-container">
      <div 
        className="code-header"
        onClick={() => toggleFunction(sectionKey)}
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
