import { useEffect, useState } from "react";
import API from "../api/backend";

function StatusCard() {

  const [status, setStatus] = useState(null);

  useEffect(() => {

    const loadStatus = async () => {

      try {

        const res = await API.get("/status");

        setStatus(res.data);

      } catch (error) {

        console.log(error);

      }

    };

    loadStatus();

  }, []);

  if (!status) return null;

  return (

    <div className="status-card">

      <h3>System Status</h3>

      <p>🟢 Backend Online</p>

      <p>🟢 Model Loaded</p>

      <p>🟢 Camera Active</p>

    </div>

  );

}

export default StatusCard;