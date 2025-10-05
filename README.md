# Universal-ECG-Viewer-with-GPU-Acceleration
Python ECG READER



Contributing to Universal ECG Viewer
Thank you for your interest in contributing!
How to Contribute
Reporting Bugs

Check if the bug has already been reported in Issues
If not, create a new issue with:

Clear description of the problem
Steps to reproduce
Expected vs actual behavior
System info (OS, Python version, GPU model if applicable)
Error messages/logs



Suggesting Features

Open a new issue with the enhancement label
Describe the feature and its benefits
Include use cases and examples if possible

Code Contributions
Setup Development Environment
bashgit clone https://github.com/yourusername/ecg-viewer-gpu.git
cd ecg-viewer-gpu
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
Making Changes

Fork the repository
Create a branch for your feature/fix:

bash   git checkout -b feature/amazing-feature

Make your changes following the code style below
Test your changes thoroughly
Commit with clear messages:

bash   git commit -m "Add amazing feature"

Push to your fork:

bash   git push origin feature/amazing-feature

Open a Pull Request with:

Clear description of changes
Reference to related issues
Screenshots/examples if applicable



Code Style Guidelines
Python Code Style

Follow PEP 8
Use meaningful variable names
Add docstrings to functions and classes
Keep functions focused and simple
Maximum line length: 100 characters

Example:
pythondef calculate_heart_rate(self, ecg_data):
    """
    Calculate heart rate from ECG signal.
    
    Args:
        ecg_data (np.ndarray): ECG signal data
        
    Returns:
        int: Heart rate in BPM or None if detection failed
    """
    # Implementation here
    pass
Comments

Write clear, concise comments
Use Bulgarian for UI strings (to match existing code)
Use English for technical comments
Explain why, not what (code should be self-explanatory)

Git Commit Messages
Format: <type>: <description>
Types:

feat: New feature
fix: Bug fix
docs: Documentation changes
style: Code style changes (formatting, no logic change)
refactor: Code refactoring
perf: Performance improvements
test: Adding tests
chore: Maintenance tasks

Examples:
feat: Add support for EDF file format
fix: Correct GPU memory leak in filtering
docs: Update installation instructions
perf: Optimize R-peak detection algorithm
Priority Areas for Contribution
High Priority

EDF/EDF+ file format support
WFDB format support
Automated arrhythmia detection
Improved GPU algorithms
Unit tests

Medium Priority

Real-time data streaming
Additional export formats (JSON, HDF5)
Heart rate variability (HRV) analysis
Multi-language UI support
Dark mode theme

Low Priority

Cloud storage integration
Mobile app version
Web-based viewer
Plugin system

Testing
Before submitting:

Test with different file formats
Test with various file sizes (small, medium, large)
Test both CPU and GPU modes (if applicable)
Verify no memory leaks
Check that existing features still work

Test Cases to Consider

Loading files with different header sizes
Files with 3, 5, 12, and custom lead counts
Different sampling rates (125Hz - 4000Hz)
Partial loading functionality
Export features (CSV, PNG, PDF)
Navigation controls
Filtering on/off
Auto-scaling

Code Review Process

Maintainers will review your PR within 5-7 days
Address any requested changes
Once approved, your PR will be merged
You'll be added to the contributors list

Questions?

Open a Discussion
Comment on relevant issues
Contact maintainers

Code of Conduct

Be respectful and constructive
Welcome newcomers
Focus on the code, not the person
Help others learn and grow

Recognition
Contributors will be:

Listed in README.md
Mentioned in release notes
Thanked publicly

Thank you for making this project better!
