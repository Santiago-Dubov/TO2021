import React, { useState, useEffect} from 'react';
import { Form, Button, Popover, OverlayTrigger} from "react-bootstrap";
import "bootstrap/dist/css/bootstrap.min.css";
import './components.css'

function Questions( ) {  
  
    const initialState = {
      questionNumber : '0'
    };

    const [fileName, setFileName] = useState("Upload Boundary File");

    const [questionNumber, setQuestionNumber] = useState(initialState);
 
    const popover = (
        <Popover id="popover-basic">
          <Popover.Title as="h3">Your answer correct! Yeayy</Popover.Title>
          <Popover.Content>
            And here's some <strong>amazing</strong> content. It's very engaging.
            right?
          </Popover.Content>
        </Popover>
      );

    const Example = () => (
       <OverlayTrigger trigger="click" placement="right" overlay={popover}>
         <Button variant="info">Check your answer here</Button>
       </OverlayTrigger>
    );  

    useEffect(() => {}, [])

    return (
        <div class="right">
            <div class="word">
                <p class="top"> <span> 1.</span> What do you think about fish?</p>
                <div class="input">
                    <Form.Control as="textarea" placeholder="Put your answer here" rows={3} />
                </div>
                <Example/>
            </div>
            <div class="submit">
                <Button variant="dark">Submit</Button>
            </div>
        </div>
    );
}

export default Questions;