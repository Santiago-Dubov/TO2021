import React, { useState, useEffect } from 'react';
import { Form, Row } from "react-bootstrap";
import "bootstrap/dist/css/bootstrap.min.css";
import './components.css'

function Import( ) {  
  
    const initialState = {
      watchHours : '0',
      carbonFootprint : '0'
    }  

    const [fileName, setFileName] = useState("Upload Boundary File");


    useEffect(() => {}, [])

    return (
        <div class="right">
            <div class="word">
                <p class="top">" Upload your pdf here</p>
                <p class="bottom">It only takes a moment :)</p>
            </div>
            <div class="input_file">
                <Form>
                    <Form.Group>
                        <Form.File id="exampleFormControlFile1" />
                    </Form.Group>
                </Form>
            </div>
        </div>
    );
}

export default Import;