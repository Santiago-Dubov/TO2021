import React, { useState, useEffect } from 'react';
import './component/components.css'
import Welcome from './component/welcome.js'
import Questions from './component/questions.js'

function Main( ) {  
  
    const initialState = {
      watchHours : '0',
      carbonFootprint : '0'
    } 

    const [state, setState] = useState(initialState)

    useEffect(() => {}, [])

    return (
        <div class="main">
            <Welcome></Welcome>
            <Questions></Questions>
        </div>
    );
}

export default Main;