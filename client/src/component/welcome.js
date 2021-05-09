import React, { useState, useEffect } from 'react';
import "bootstrap/dist/css/bootstrap.min.css";
import './components.css'
import background from '../img/background.png'


function Welcome( ) {  
  
    const initialState = {
      watchHours : '0',
      carbonFootprint : '0'
    } 

    const [fileName, setFileName] = useState("Upload Boundary File");

 
    useEffect(() => {}, [])

    return (
        <div class="left">
            <div class="welcoming_word">
                <h1 style={{fontSize:'60px'}}>Welcome </h1>
                <h1 style={{fontSize:'60px'}}>to QuizMe!</h1>
            </div>
            <img src={background}></img>
        </div>
    );
}

export default Welcome;