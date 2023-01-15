
import Button from "@mui/material/Button"
import Card from "@mui/material/Card"
import TextField from "@mui/material/TextField"
import Stack from "@mui/material/Stack"
import ListItem from "@mui/material/ListItem"
import { useState } from "react";

const Register = () => {
  const [currentPassword, setCurrentPassword] = useState("");
  const [currentUsername, setCurrentUsername] = useState("");

  const handleRegister = () => {
    const requestOptions = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        username: currentUsername,
        password: currentPassword,
      }),
    };

    const url = "/api/auth/register";
    fetch(url, requestOptions).then((response) => response.text()).then((token) => {
      localStorage.setItem('apiToken', token)
    })
  };

  return (
    <Card>
      <Stack>
        <ListItem>
          <TextField
            id="outlined-username-input"
            label="Username"
            onChange={(e) => setCurrentUsername(e.target.value)}
          >
            User
          </TextField>
        </ListItem>
        <ListItem>
          <TextField
            type="password"
            id="outlined-password-input"
            label="Password"
            onChange={(e) => setCurrentPassword(e.target.value)}
          >
            Password
          </TextField>
        </ListItem>
        <ListItem>
          <Button onClick={handleRegister}>Register</Button>
        </ListItem>
      </Stack>
    </Card>
  );
};

export default Register;
