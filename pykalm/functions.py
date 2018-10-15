import numpy as np


def mymatmul(matrix_list: list) -> np.ndarray:
	"""
	Custom function to multiply matrices starting with the furthest right-hand operations.
	:param matrix_list:
	:return:
	"""
	result = None
	for matrix in reversed(matrix_list):
		if result:
			result = np.matmul(result, matrix)
		else:
			result = matrix
	return result


def propagate_state(current_state, transition_matrix, state_covariance=None) -> tuple:
	"""
	Propagates from current state and current state covariance to the next according to the linear relation defined by
	the transition matrix.
	:param current_state:
	:param transition_matrix:
	:param state_covariance:
	:return:
	"""
	new_predicted_state = np.matmul(transition_matrix, current_state)
	if state_covariance:
		new_state_covariance = mymatmul([transition_matrix, state_covariance, transition_matrix.T])
		return new_predicted_state, new_state_covariance
	else:
		new_state_covariance = np.eye(max(current_state.state))
		return new_predicted_state, new_state_covariance


def predict_state(propagated_state, gain_matrix, observations, obs_information_matrix) -> np.ndarray:
	"""
	Calculates the predicted state according to the observations and propagated state.
	:param propagated_state:
	:param gain_matrix:
	:param observations:
	:param obs_information_matrix:
	:return:
	"""
	return propagated_state + np.matmul(gain_matrix, observations - np.matmul(obs_information_matrix, propagated_state))


def prediction_covariance(gain_matrix, obs_information_matrix, obs_covariance) -> np.ndarray:
	"""
	Calculates the propagated covariance of the predicted state according to the Kalman prediction.
	:param gain_matrix:
	:param obs_information_matrix:
	:param obs_covariance:
	:return:
	"""
	return np.matmul(np.eye(obs_covariance.shape[0] - np.matmul(gain_matrix, obs_information_matrix)), obs_covariance)


def gain_matrix(obs_information_matrix, observation_covariance=None, state_covariance=None) -> np.ndarray:
	"""
	Carries out the determination of the Kalman gain matrix according to the properties of the Penrose-Moore pseudo
	inverse.
	:param obs_information_matrix:
	:param observation_covariance:
	:param state_covariance:
	:return:
	"""
	if not observation_covariance:
		observation_covariance = np.eye(obs_information_matrix.shape[1])
	if not state_covariance:
		state_covariance = np.eye(obs_information_matrix.shape[0])
	return mymatmul(
		[
			state_covariance, obs_information_matrix.T,
			np.linalg.inv(
				mymatmul([
					obs_information_matrix,
					state_covariance,
					obs_information_matrix.T]) +
				observation_covariance
			)
		]
	)
