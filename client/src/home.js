import React, { useState, useEffect } from 'react';
import './component/components.css'
import Import from './component/import.js'
import Welcome from './component/welcome.js'

function Home( ) {  
  
    const initialState = {
      watchHours : '0',
      carbonFootprint : '0'
    } 

    const [state, setState] = useState(initialState)

    useEffect(() => {}, [])

    return (
        <div class="main">
            <Welcome></Welcome>
            <Import></Import>
        </div>
    );
}

export default Home;