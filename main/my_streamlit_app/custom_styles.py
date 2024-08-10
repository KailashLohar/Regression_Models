# custom_styles.py

def load_custom_styles():
    return """
    <style>
    .title-box {
        background-color: #90a4ae;
        border-radius: 20px;
        padding: 0px;
        margin-top: -50px; /* Pushes the box up */
        margin-bottom: 0px; /* Reduced margin */
        text-align: center;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    h2 {
        margin-top: -20px; /* Further reduced gap */
        margin-bottom: -20px; /* Further reduced gap */
    }
    .small-button {
        display: inline-block;
        padding: 0.25em 1em;
        font-size: 0.75em;
        margin: 0.25em;
        cursor: pointer;
        text-align: center;
        text-decoration: none;
        color: #FFF;
        background-color: #007BFF;
        border-radius: 0.25em;
        border: none;
    }
    .help-section {
        position: absolute;
        top: -40px;
        right: 50px;
        z-index: 9999;
    }
    </style>
    """