let socket = null;

export const connectWebSocket = (onMessage) => {

  socket = new WebSocket(
    "ws://127.0.0.1:8000/ws"
  );

  socket.onopen = () => {
    console.log("WebSocket Connected");
  };

  socket.onmessage = (event) => {

    const data = JSON.parse(event.data);

    onMessage(data);
  };

  socket.onclose = () => {
    console.log("WebSocket Closed");
  };

  return socket;
};

export const getSocket = () => socket;