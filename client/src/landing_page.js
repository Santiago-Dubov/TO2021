import React, { useState, useEffect } from 'react';
import { Col, Row, Container, Button} from "react-bootstrap";
import './component/components.css'
import revolution from './img/revolution.png'
import logo from './img/logo.jpg'

function LandingPage( ) {  
  
    const initialState = {
      watchHours : '0',
      carbonFootprint : '0'
    } 

    const [state, setState] = useState(initialState)

    useEffect(() => {}, [])

    return (
        <Container fluid>
            <Row>
                <Col style={{marginTop:'20px', marginLeft:'30rem',}} >
                    <img style={{width:'60px',height:'60px', marginRight:'200px'}} src={logo}></img>
                    <Button style={{margin:'20px'}} variant="light">About</Button>
                    <Button style={{margin:'20px'}} variant="light">Collections</Button>
                    <Button style={{margin:'20px'}} variant="light">Contribute</Button>
                    <Button style={{margin:'20px'}} variant="dark">Join QuizMe</Button>
                </Col>
            </Row>
            <Row>
                <Col>
                    <div class="lp">
                        <h1>REVOLUTIONIZE your study habits</h1>
                        <p>An app by students for students</p>
                        <Button style={{marginTop:'20px'}} variant="info" block>Create a new account</Button>
                        <Button style={{marginTop:'20px'}} variant="outline-info" block>Sign in to an existing account</Button>
                    </div>
                </Col>
                <Col>
                    <img style={{marginTop:'12rem', width:'20rem', height:'20rem'}} src={revolution}></img>
                </Col>
            </Row>
        </Container>
    );
}

export default LandingPage;